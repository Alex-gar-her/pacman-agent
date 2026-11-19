# my_team.py
# ---------------
# Eutopia Pacman Contest - equipo autocontenido.
# Solo usa: util (Queue, Counter), CaptureAgent y la logica de los labs
# (BFS de search.py, funciones de evaluacion lineales tipo ReflexAgent).

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]


# Memoria compartida entre los dos agentes (se reinicia en cada partida).
SHARED = {}


##########
# Agents #
##########

class TeamAgent(CaptureAgent):
    """
    Base comun: tabla de distancias reales (BFS), celdas de frontera,
    callejones sin salida, memoria de invasores y las dos politicas
    (ataque y defensa) que usan ambos agentes segun el rol dinamico.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.patrol_target = None

    # ------------------------------------------------------------------
    # Precalculo inicial (15 s)
    # ------------------------------------------------------------------
    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        self.start = game_state.get_agent_position(self.index)
        self.walls = game_state.get_walls()
        self.width = self.walls.width
        self.height = self.walls.height
        self.team_indices = self.get_team(game_state)
        SHARED.clear()
        SHARED['pos'] = {}

        self.cells = [(x, y) for x in range(self.width) for y in range(self.height)
                      if not self.walls[x][y]]
        self.cell_set = set(self.cells)
        self.neighbors = {}
        for (x, y) in self.cells:
            nbs = []
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (x + dx, y + dy)
                if n in self.cell_set:
                    nbs.append(n)
            self.neighbors[(x, y)] = nbs

        # Distancias reales entre todos los pares (BFS desde cada celda)
        self.dist_table = {}
        for c in self.cells:
            self.dist_table[c] = self.bfs_distances(c)

        # Frontera: celdas de nuestro lado pegadas a la linea central
        mid = self.width // 2
        self.mid = mid
        if self.red:
            home_x, other_x, inward = mid - 1, mid, -1
        else:
            home_x, other_x, inward = mid, mid - 1, 1
        self.home_cells = [c for c in self.cells
                           if c[0] == home_x and (other_x, c[1]) in self.cell_set]
        if not self.home_cells:
            self.home_cells = [c for c in self.cells if c[0] == home_x]
        # Puntos de patrulla: 2 columnas dentro de nuestro campo
        self.patrol_cells = [c for c in self.cells if c[0] == home_x + 2 * inward]
        if not self.patrol_cells:
            self.patrol_cells = list(self.home_cells)

        self.compute_dead_ends()

    def bfs_distances(self, source):
        """BFS de search.py (cola FIFO) pero expandiendo todo el laberinto."""
        queue = util.Queue()
        queue.push(source)
        dist = {source: 0}
        while not queue.is_empty():
            cur = queue.pop()
            for nb in self.neighbors[cur]:
                if nb not in dist:
                    dist[nb] = dist[cur] + 1
                    queue.push(nb)
        return dist

    def compute_dead_ends(self):
        """
        Poda iterativa de celdas con un unico vecino. dead_len[c] = longitud
        del callejon en el que esta la celda c (0 si no esta en uno).
        """
        degree = {c: len(self.neighbors[c]) for c in self.cells}
        removed = set()
        order = []
        nxt = {}
        depth = {}
        queue = util.Queue()
        for c in self.cells:
            if degree[c] == 1:
                depth[c] = 1
                queue.push(c)
        while not queue.is_empty():
            c = queue.pop()
            removed.add(c)
            order.append(c)
            rem = [n for n in self.neighbors[c] if n not in removed]
            if len(rem) == 1:
                n = rem[0]
                nxt[c] = n
                degree[n] -= 1
                if degree[n] == 1 and n not in depth:
                    depth[n] = depth[c] + 1
                    queue.push(n)
        self.dead_len = {}
        for c in reversed(order):
            n = nxt.get(c)
            if n is not None and n in depth and n in self.dead_len:
                self.dead_len[c] = self.dead_len[n]
            else:
                self.dead_len[c] = depth[c]

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------
    def dist(self, a, b):
        try:
            return self.dist_table[a][b]
        except KeyError:
            return self.get_maze_distance(a, b)

    def min_dist(self, pos, targets):
        if not targets:
            return None
        return min(self.dist(pos, t) for t in targets)

    @staticmethod
    def pos_of(state):
        p = state.get_position()
        return (int(round(p[0])), int(round(p[1])))

    def time_left(self, game_state):
        return getattr(game_state.data, 'timeleft', 1200)

    def on_my_side(self, pos):
        return pos[0] < self.mid if self.red else pos[0] >= self.mid

    def home_distance(self, pos):
        if self.on_my_side(pos):
            return 0
        return self.min_dist(pos, self.home_cells)

    def get_successor(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        return successor

    def visible_enemies(self, game_state):
        res = []
        for i in self.get_opponents(game_state):
            st = game_state.get_agent_state(i)
            p = st.get_position()
            if p is not None:
                res.append((st, (int(round(p[0])), int(round(p[1])))))
        return res

    def threat_ghosts(self, game_state):
        return [p for st, p in self.visible_enemies(game_state)
                if not st.is_pacman and st.scared_timer <= 2]

    def visible_invaders(self, game_state):
        return [p for st, p in self.visible_enemies(game_state) if st.is_pacman]

    def mate_position(self):
        for i in self.team_indices:
            if i != self.index:
                return SHARED['pos'].get(i)
        return None

    # ------------------------------------------------------------------
    # Memoria: rastreo de invasores por comida desaparecida
    # ------------------------------------------------------------------
    def update_memory(self, game_state):
        now = self.time_left(game_state)
        my_pos = self.pos_of(game_state.get_agent_state(self.index))
        SHARED['pos'][self.index] = my_pos

        invaders = self.visible_invaders(game_state)
        if invaders:
            nearest = min(invaders, key=lambda p: self.dist(my_pos, p))
            SHARED['invader'] = (nearest, now)
            return

        prev = self.get_previous_observation()
        if prev is not None:
            before = self.get_food_you_are_defending(prev).as_list()
            after = set(self.get_food_you_are_defending(game_state).as_list())
            missing = [f for f in before if f not in after]
            if missing:
                m = min(missing, key=lambda f: self.dist(my_pos, f))
                SHARED['invader'] = (m, now)

        inv = SHARED.get('invader')
        if inv is not None:
            ipos, t = inv
            manhattan = abs(ipos[0] - my_pos[0]) + abs(ipos[1] - my_pos[1])
            # Si estamos a <=5 y no lo vemos, ya no esta ahi; o si es muy antiguo
            if manhattan <= 5 or now - t > 60:
                del SHARED['invader']

    # ------------------------------------------------------------------
    # Politica ofensiva
    # ------------------------------------------------------------------
    def food_target_cost(self, pos, foods, ghosts, mate_pos):
        if not foods:
            return 0
        best = None
        for f in foods:
            d = self.dist(pos, f)
            c = d
            if ghosts:
                gd = self.min_dist(f, ghosts)
                if gd <= d + 1:
                    c += 8
                if self.dead_len.get(f, 0) and gd <= d + 6:
                    c += 6
            if mate_pos is not None and self.dist(mate_pos, f) < d:
                c += 6
            if best is None or c < best:
                best = c
        return best

    def offense_action(self, game_state, force_home=False):
        now = self.time_left(game_state)
        actions = game_state.get_legal_actions(self.index)
        my_state = game_state.get_agent_state(self.index)
        my_pos = self.pos_of(my_state)
        carrying = my_state.num_carrying
        food_list = self.get_food(game_state).as_list()
        capsules = self.get_capsules(game_state)
        ghosts = self.threat_ghosts(game_state)
        mate_pos = self.mate_position()

        danger_now = None
        if ghosts and my_state.is_pacman:
            danger_now = self.min_dist(my_pos, ghosts)

        # ---- decision: volver a casa o seguir comiendo ----
        until = SHARED.get('scared_until')
        scared_active = until is not None and now > until
        limit = 8 if scared_active else 5
        moves_left = now // 4
        home_now = self.home_distance(my_pos)
        near_food = self.min_dist(my_pos, food_list) if food_list else 0

        return_mode = force_home
        if carrying > 0 and not return_mode:
            if carrying >= limit:
                return_mode = True
            elif danger_now is not None and danger_now <= 6:
                return_mode = True
            elif home_now + 4 >= moves_left:
                return_mode = True
            elif len(food_list) <= 2:
                return_mode = True
            elif carrying >= 2 and near_food > home_now + 4:
                return_mode = True

        rev = Directions.REVERSE[my_state.configuration.direction]
        best_val = None
        best_actions = []
        for a in actions:
            succ = self.get_successor(game_state, a)
            s_state = succ.get_agent_state(self.index)
            pos = self.pos_of(s_state)
            v = 0.0

            if a == Directions.STOP:
                v -= 30
            elif a == rev:
                v -= 3
            # Si el sucesor es nuestro punto de inicio estando lejos: nos han comido
            if pos == self.start and self.dist(my_pos, self.start) > 1:
                v -= 2000

            # Peligro: fantasmas visibles
            if s_state.is_pacman and ghosts:
                d = self.min_dist(pos, ghosts)
                if d <= 1:
                    v -= 1000
                elif d <= 5:
                    v -= (6 - d) ** 2 * 8
                dl = self.dead_len.get(pos, 0)
                if dl and d <= 6:
                    v -= 60 + 15 * dl

            food_after = self.get_food(succ).as_list()
            if return_mode:
                v -= 12 * self.home_distance(pos)
                v += 15 * (len(food_list) - len(food_after))
                if ghosts and pos in capsules:
                    v += 60
            else:
                v -= 100 * len(food_after)
                v -= 1.5 * self.food_target_cost(pos, food_after, ghosts, mate_pos)
                if pos in capsules:
                    v += 60 if (danger_now is not None and danger_now <= 8) else 20
                elif danger_now is not None and danger_now <= 8 and capsules:
                    v -= 4 * self.min_dist(pos, capsules)

            if best_val is None or v > best_val:
                best_val = v
                best_actions = [a]
            elif v == best_val:
                best_actions.append(a)

        action = random.choice(best_actions)
        # Registrar captura de capsula (enemigos asustados ~40 movimientos suyos)
        nxt = self.pos_of(self.get_successor(game_state, action).get_agent_state(self.index))
        if nxt in capsules:
            SHARED['scared_until'] = now - 150
        return action

    # ------------------------------------------------------------------
    # Politica defensiva
    # ------------------------------------------------------------------
    def choose_patrol_target(self, game_state, my_pos):
        food = self.get_food_you_are_defending(game_state).as_list()
        if not food:
            return self.start
        cy = sum(f[1] for f in food) / float(len(food))
        cands = sorted(self.patrol_cells, key=lambda c: abs(c[1] - cy))[:3]
        if (self.patrol_target is None or self.patrol_target not in cands
                or self.dist(my_pos, self.patrol_target) <= 1):
            others = [c for c in cands if c != self.patrol_target] or cands
            self.patrol_target = random.choice(others)
        return self.patrol_target

    def defense_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        my_state = game_state.get_agent_state(self.index)
        my_pos = self.pos_of(my_state)
        scared = my_state.scared_timer > 0
        invaders = self.visible_invaders(game_state)

        if invaders:
            target = min(invaders, key=lambda p: self.dist(my_pos, p))
            mode = 'visible'
        elif SHARED.get('invader') is not None:
            target = SHARED['invader'][0]
            mode = 'estimated'
        else:
            target = self.choose_patrol_target(game_state, my_pos)
            mode = 'patrol'

        rev = Directions.REVERSE[my_state.configuration.direction]
        best_val = None
        best_actions = []
        for a in actions:
            succ = self.get_successor(game_state, a)
            s_state = succ.get_agent_state(self.index)
            pos = self.pos_of(s_state)
            v = 0.0
            if s_state.is_pacman:
                v -= 500
            if a == Directions.STOP:
                v -= 5
            elif a == rev:
                v -= 2
            d = self.dist(pos, target)
            if mode == 'visible':
                if scared:
                    # Bloquear sin morir: mantenerse a distancia 2
                    if d <= 1:
                        v -= 1000
                    v -= 10 * abs(d - 2)
                else:
                    v -= 12 * d
            elif mode == 'estimated':
                v -= 8 * d
            else:
                v -= 6 * d
            if best_val is None or v > best_val:
                best_val = v
                best_actions = [a]
            elif v == best_val:
                best_actions.append(a)
        return random.choice(best_actions)


class OffensiveReflexAgent(TeamAgent):
    """Atacante: come, evita fantasmas/callejones, vuelve a depositar."""

    def choose_action(self, game_state):
        self.update_memory(game_state)
        my_state = game_state.get_agent_state(self.index)
        my_pos = self.pos_of(my_state)
        # Si estamos en casa, sin comida, y un invasor esta muy cerca: ayudar
        if (not my_state.is_pacman and my_state.scared_timer == 0
                and my_state.num_carrying == 0):
            invaders = self.visible_invaders(game_state)
            if invaders and self.min_dist(my_pos, invaders) <= 3:
                return self.defense_action(game_state)
        return self.offense_action(game_state)


class DefensiveReflexAgent(TeamAgent):
    """Defensor: patrulla, rastrea invasores; ataca si vamos perdiendo y no hay amenaza."""

    def choose_action(self, game_state):
        self.update_memory(game_state)
        my_state = game_state.get_agent_state(self.index)
        has_signal = bool(self.visible_invaders(game_state)) or SHARED.get('invader') is not None
        behind = self.get_score(game_state) < 0
        carrying = my_state.num_carrying

        if my_state.is_pacman and carrying > 0:
            # Ya estamos dentro con comida: asegurar los puntos
            return self.offense_action(game_state, force_home=has_signal)
        if behind and not has_signal:
            return self.offense_action(game_state)
        return self.defense_action(game_state)