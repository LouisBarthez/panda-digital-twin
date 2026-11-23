// jumeau_robot : exécute sur le vrai Panda la trajectoire validée dans le jumeau MuJoCo.
// libfranka 0.8.0 (firmware 4.1.1). Appelé par serveur_robot.py, utilisable aussi à la main :
//
//   ./examples/jumeau_robot 192.168.153.114 etat
//        -> une ligne JSON : {"q":[7], "O_T_EE":[16], "largeur":w}   (lecture seule)
//   ./examples/jumeau_robot 192.168.153.114 executer fichier_etapes.txt
//
// Fichier d'étapes (une par ligne) :
//   vitesse 0.1                      facteur de vitesse du MotionGenerator (0.02 à 0.3)
//   depart q1 .. q7                  pose de départ utilisée par la simulation (contrôle)
//   q nom q1 .. q7                   mouvement articulaire synchronisé jusqu'à ces angles
//   ouvrir largeur                   pince : ouverture (m)
//   saisir largeur force             pince : prise (m, N) ; échec -> on rouvre et on remonte
//
// Pendant les mouvements, une ligne "Q q1 .. q7" est écrite toutes les 40 ms (angles mesurés) : le
// serveur la transmet au PC Windows, qui fait suivre le bras virtuel en direct (ombre numérique).
// La boucle à 1 kHz ne fait que ranger les angles dans des variables atomiques ; l'écriture se fait
// dans un autre thread, pour ne jamais ralentir la boucle temps réel.
//
// Codes de sortie : 0 = OK, 1 = erreur libfranka (réflexe, etc.), 2 = robot pas dans la pose de
// départ, 3 = prise ratée (bras remonté au point précédent), 4 = fichier invalide.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <franka/exception.h>
#include <franka/gripper.h>
#include <franka/robot.h>

#include "examples_common.h"

namespace {

struct Etape {
  std::string type, nom;
  std::array<double, 7> q{};
  double a = 0.0, b = 0.0;
};

std::array<std::atomic<double>, 7> g_q;   // derniers angles mesurés (écrits par la boucle 1 kHz)
std::atomic<bool> g_fin{false};
std::mutex g_sortie;                      // une ligne à la fois sur la sortie standard

void ecrire(const std::string& ligne) {
  std::lock_guard<std::mutex> l(g_sortie);
  std::cout << ligne << std::endl;
}

// thread qui écrit "Q ..." toutes les 40 ms ; arrêté et attendu à la destruction (même sur exception)
struct Imprimeur {
  std::thread t;
  Imprimeur() {
    t = std::thread([] {
      while (!g_fin) {
        std::ostringstream o;
        o.precision(5);
        o << "Q";
        for (auto& v : g_q) o << " " << v.load(std::memory_order_relaxed);
        ecrire(o.str());
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
      }
    });
  }
  ~Imprimeur() {
    g_fin = true;
    if (t.joinable()) t.join();
  }
};

template <size_t N>
std::string json_tab(const std::array<double, N>& v) {
  std::ostringstream o;
  o.precision(6);
  o << "[";
  for (size_t i = 0; i < N; i++) o << (i ? "," : "") << v[i];
  o << "]";
  return o.str();
}

double largeur_pince(franka::Gripper* g) {
  if (!g) return -1.0;
  try {
    return g->readOnce().width;
  } catch (const franka::Exception&) {
    return -1.0;
  }
}

void mesure(const std::string& nom, franka::Robot& robot, franka::Gripper* g) {
  franka::RobotState s = robot.readOnce();
  for (size_t i = 0; i < 7; i++) g_q[i].store(s.q[i]);
  std::ostringstream o;
  o << "MESURE {\"nom\":\"" << nom << "\",\"q\":" << json_tab(s.q) << ",\"O_T_EE\":" << json_tab(s.O_T_EE)
    << ",\"largeur\":" << largeur_pince(g) << "}";
  ecrire(o.str());
}

// mouvement articulaire (MotionGenerator de libfranka) en rangeant les angles mesurés au passage
void bouger(franka::Robot& robot, double vitesse, const std::array<double, 7>& but) {
  MotionGenerator mg(vitesse, but);
  robot.control([&mg](const franka::RobotState& s, franka::Duration d) -> franka::JointPositions {
    for (size_t i = 0; i < 7; i++) g_q[i].store(s.q[i], std::memory_order_relaxed);
    return mg(s, d);
  });
}

bool lire_q(std::istringstream& is, std::array<double, 7>& q) {
  for (auto& v : q)
    if (!(is >> v)) return false;
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage : " << argv[0] << " <ip-robot> etat | executer <fichier>" << std::endl;
    return 4;
  }
  const std::string ip = argv[1], mode = argv[2];
  try {
    franka::Robot robot(ip);

    // ------------------------------------------------------------ lecture seule
    if (mode == "etat") {
      franka::RobotState s = robot.readOnce();
      double w = -1.0;
      try {
        franka::Gripper g(ip);
        w = g.readOnce().width;
      } catch (const franka::Exception&) {
      }
      std::cout << "{\"q\":" << json_tab(s.q) << ",\"O_T_EE\":" << json_tab(s.O_T_EE)
                << ",\"largeur\":" << w << "}" << std::endl;
      return 0;
    }
    if (mode != "executer" || argc < 4) {
      std::cerr << "mode inconnu" << std::endl;
      return 4;
    }

    // ------------------------------------------------------------ lecture du fichier
    std::ifstream f(argv[3]);
    if (!f) {
      std::cout << "ERREUR fichier illisible : " << argv[3] << std::endl;
      return 4;
    }
    double vitesse = 0.1;
    std::array<double, 7> depart{};
    bool a_depart = false;
    std::vector<Etape> etapes;
    std::string ligne;
    int num = 0;
    while (std::getline(f, ligne)) {
      num++;
      std::istringstream is(ligne);
      std::string t;
      if (!(is >> t)) continue;
      Etape e;
      e.type = t;
      bool ok = true;
      if (t == "vitesse") {
        ok = static_cast<bool>(is >> vitesse);
        if (ok) continue;
      } else if (t == "depart") {
        ok = lire_q(is, depart);
        a_depart = ok;
        if (ok) continue;
      } else if (t == "q") {
        ok = (is >> e.nom) && lire_q(is, e.q);
      } else if (t == "ouvrir") {
        ok = static_cast<bool>(is >> e.a);
        e.nom = "ouvrir";
      } else if (t == "saisir") {
        ok = (is >> e.a) && (is >> e.b);
        e.nom = "saisir";
      } else {
        ok = false;
      }
      if (!ok) {
        std::cout << "ERREUR ligne " << num << " illisible : " << ligne << std::endl;
        return 4;
      }
      etapes.push_back(e);
    }
    if (!a_depart) {
      std::cout << "ERREUR pas de ligne 'depart'" << std::endl;
      return 4;
    }
    vitesse = std::min(std::max(vitesse, 0.02), 0.3);

    // ------------------------------------------------------------ le robot est-il là où la sim l'a pris ?
    franka::RobotState s0 = robot.readOnce();
    double ecart = 0.0;
    for (size_t i = 0; i < 7; i++) ecart = std::max(ecart, std::abs(s0.q[i] - depart[i]));
    if (ecart > 0.03) {
      std::cout << "ERREUR le robot n'est plus dans la pose de depart (ecart " << ecart
                << " rad) : refaire la simulation (E)" << std::endl;
      return 2;
    }

    franka::Gripper gripper(ip);
    setDefaultBehavior(robot);
    // mêmes seuils de collision que les exemples libfranka (assez bas)
    robot.setCollisionBehavior(
        {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
        {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
        {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}},
        {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}});

    std::vector<std::array<double, 7>> visites{s0.q};
    for (size_t i = 0; i < 7; i++) g_q[i].store(s0.q[i]);
    Imprimeur imprimeur;
    mesure("depart", robot, &gripper);
    for (size_t k = 0; k < etapes.size(); k++) {
      const Etape& e = etapes[k];
      ecrire("ETAPE " + std::to_string(k + 1) + "/" + std::to_string(etapes.size()) + " " + e.type + " " + e.nom);
      if (e.type == "q") {
        bouger(robot, vitesse, e.q);
        visites.push_back(e.q);
        mesure(e.nom, robot, &gripper);
      } else if (e.type == "ouvrir") {
        gripper.move(e.a, 0.05);
        ecrire("PINCE " + std::to_string(largeur_pince(&gripper)));
      } else if (e.type == "saisir") {
        bool ok = gripper.grasp(e.a, 0.03, e.b, 0.008, 0.008);
        double w = largeur_pince(&gripper);
        ecrire("PINCE " + std::to_string(w));
        mesure("prise", robot, &gripper);
        if (!ok) {
          ecrire("ECHEC_PRISE largeur " + std::to_string(w * 1000) + " mm : on rouvre et on remonte");
          gripper.move(0.08, 0.05);
          ecrire("PINCE " + std::to_string(largeur_pince(&gripper)));
          if (visites.size() >= 2) bouger(robot, vitesse, visites[visites.size() - 2]);
          return 3;
        }
      }
    }
    ecrire("FIN OK");
    return 0;
  } catch (const franka::Exception& ex) {
    ecrire(std::string("ERREUR libfranka : ") + ex.what());
    return 1;
  }
}
