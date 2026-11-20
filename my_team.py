# my_team.py  (compact version)
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions, Actions


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    Returns a list of two agents that form the team (first: attacker, second: defender).
    """
    return [eval(first)(first_index), eval(second)(second_index)]


TOTAL_MOVES = 300  # Moves per agent (1200 moves / 4 agents)
MOVES = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]


###############################################################
# Code reused from our labs: search.py / search_agents.py     #
###############################################################

class SearchProblem:
    """Abstract search problem (copied from our search.py)."""

    def get_start_state(self):
        util.raise_not_defined()

    def is_goal_state(self, state):
        util.raise_not_defined()

    def get_successors(self, state):
        util.raise_not_defined()

    def get_cost_of_actions(self, actions):
        util.raise_not_defined()


def null_heuristic(state, problem=None):
    """Trivial heuristic (copied from our search.py)."""
    return 0


def a_star_search(problem, heuristic=null_heuristic):
    """Search the node that has the lowest combined cost and heuristic first. (Copied from our search.py)"""
    priority_queue = util.PriorityQueue()  # Inicializes the data structure (Priority Queue)
    visited_states = set()  # List of visited states
    start_state = problem.get_start_state()  # Get the inicial state on start_state
    start_cost = 0  # Set the inicial cost in 0 [g(x)]
    start_heuristic = heuristic(start_state, problem)  # Compute the heuristic value on the start [h(x)]
    start_priority = start_cost + start_heuristic  # Compute the priority value at the start --> f(x) = g(x) + h(x)
    priority_queue.push((start_state, [], start_cost), start_priority)

    while not priority_queue.is_empty():
        current_state, actions, current_cost = priority_queue.pop()  # Take out the element with less cost in the queue

        # If the state is the final goal, we return the actions to arrive there
        if problem.is_goal_state(current_state):
            return actions

        # If haven't visited the current state, add it to the list of visited states
        if current_state not in visited_states:
            visited_states.add(current_state)

            # For every successor of the current state, check if is a new one (not visited), and if so, compute all necesary things
            for successor, action, step_cost in problem.get_successors(current_state):
                if successor not in visited_states:
                    new_actions = actions + [action]  # Compute new actions
                    new_cost = current_cost + step_cost  # Compute new acumulative cost
                    heuristic_value = heuristic(successor, problem)  # Compute the heuristic value in the successor
                    new_priority = new_cost + heuristic_value  # Compute the priority value at the correct state
                    priority_queue.push((successor, new_actions, new_cost), new_priority)


class PositionSearchProblem(SearchProblem):
    """
    The state space consists of (x,y) positions (adapted from our search_agents.py: it receives the walls).
    """

    def __init__(self, walls, start, goal, cost_fn=lambda x: 1):
        self.walls = walls
        self.startState = start
        self.goal = goal
        self.cost_fn = cost_fn

    def get_start_state(self):
        return self.startState

    def is_goal_state(self, state):
        return state == self.goal

    def get_successors(self, state):
        successors = []
        for action in MOVES:
            x, y = state
            dx, dy = Actions.direction_to_vector(action)
            next_x, next_y = int(x + dx), int(y + dy)
            if not self.walls[next_x][next_y]:
                next_state = (next_x, next_y)
                successors.append((next_state, action, self.cost_fn(next_state)))
        return successors

    def get_cost_of_actions(self, actions):
        x, y = self.get_start_state()
        cost = 0
        for action in actions:
            dx, dy = Actions.direction_to_vector(action)
            x, y = int(x + dx), int(y + dy)
            cost += self.cost_fn((x, y))
        return cost


class GoalSearchProblem(PositionSearchProblem):
    """Like AnyFoodSearchProblem: several goals, and optionally only some cells are allowed."""

    def __init__(self, walls, start, goals, cost_fn, allowed=None):
        PositionSearchProblem.__init__(self, walls, start, None, cost_fn)
        self.goals = set(goals)
        self.allowed = allowed

    def is_goal_state(self, state):
        return state in self.goals

    def get_successors(self, state):
        successors = PositionSearchProblem.get_successors(self, state)
        if self.allowed is None:
            return successors
        return [s for s in successors if self.allowed(s[0])]


def table_heuristic(state, problem):
    """Real maze distance to the closest goal (precomputed table): admissible, since every step costs >= 1."""
    return problem.table.get(state, 0)


def to_cell(pos):
    return (int(pos[0] + 0.5), int(pos[1] + 0.5))


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    Base class: precomputes the maze (real distances with BFS, dead-ends, border entries), tracks the
    observations and contains the offensive and defensive behaviours (so the two agents can swap roles).
    """

    maze_cache = {}

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.reset_memory()

    def reset_memory(self):
        self.turns = 0
        self.returning = False
        self.invader = None  # (estimated position of an invader, turn)
        self.last_ghost_turn = -100
        self.last_invader_turn = 0
        self.ghost_mem = {}
        self.patrol_points = []
        self.patrol_index = 0

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        self.reset_memory()
        self.walls = game_state.get_walls()
        self.mid = self.walls.width // 2
        try:
            self.is_red_team = bool(self.red)
        except AttributeError:
            self.is_red_team = (self.index % 2 == 0)

        self.walkable = [(x, y) for x in range(self.walls.width) for y in range(self.walls.height)
                         if not self.walls[x][y]]
        self.neighbors = {}
        for (x, y) in self.walkable:
            self.neighbors[(x, y)] = [(x + dx, y + dy) for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0))
                                      if not self.walls[x + dx][y + dy]]
        # Real distances between all the pairs of cells and dead-ends (computed once per layout)
        key = tuple(self.walkable)
        if key not in ReflexCaptureAgent.maze_cache:
            ReflexCaptureAgent.maze_cache.clear()
            tables = dict((cell, self.bfs_distances(cell)) for cell in self.walkable)
            ReflexCaptureAgent.maze_cache[key] = (tables, self.compute_dead_ends())
        self.dist_cache, self.dead_end = ReflexCaptureAgent.maze_cache[key]
        # Entries of our side (border cells that connect both sides) and distance of every cell to them
        my_x = self.mid - 1 if self.is_red_team else self.mid
        their_x = self.mid if self.is_red_team else self.mid - 1
        self.home_entries = [(my_x, y) for y in range(self.walls.height)
                             if (my_x, y) in self.neighbors and (their_x, y) in self.neighbors]
        self.home_dist = dict((c, min(self.dist_cache[c][e] for e in self.home_entries)) for c in self.walkable)

    def bfs_distances(self, source):
        """Breadth first search (as in our search.py) covering the whole maze."""
        distances = {source: 0}
        queue = util.Queue()
        queue.push(source)
        while not queue.is_empty():
            current = queue.pop()
            for successor in self.neighbors[current]:
                if successor not in distances:
                    distances[successor] = distances[current] + 1
                    queue.push(successor)
        return distances

    def compute_dead_ends(self):
        """Peels the maze (cells with one neighbour left): dead_end[cell] = depth of the dead-end corridor."""
        degree = dict((c, len(self.neighbors[c])) for c in self.walkable)
        depth, best = {}, {}
        queue = util.Queue()
        for cell in self.walkable:
            if degree[cell] <= 1:
                depth[cell] = 1
                queue.push(cell)
        while not queue.is_empty():
            current = queue.pop()
            for n in self.neighbors[current]:
                if n in depth:
                    continue
                degree[n] -= 1
                best[n] = max(best.get(n, 0), depth[current])
                if degree[n] <= 1:
                    depth[n] = best[n] + 1
                    queue.push(n)
        return depth

    ########################
    # Helpers              #
    ########################

    def dist(self, a, b):
        return self.dist_cache[a][b]

    def is_home(self, cell):
        return cell[0] < self.mid if self.is_red_team else cell[0] >= self.mid

    def carried(self, state):
        return getattr(state, 'num_carrying', getattr(state, 'numCarrying', 0))

    def mate(self, game_state):
        return [i for i in self.get_team(game_state) if i != self.index][0]

    def search(self, pos, goals, cost_fn=None, home_only=False):
        """A* to the closest goal. Returns (actions, cost) or (None, None)."""
        goals = list(goals)
        if goals == self.home_entries:
            table = self.home_dist
        else:
            table = dict((c, min(self.dist(c, g) for g in goals)) for c in self.walkable)
        problem = GoalSearchProblem(self.walls, pos, goals, cost_fn or (lambda c: 1),
                                    self.is_home if home_only else None)
        problem.table = table
        actions = a_star_search(problem, table_heuristic)
        if not actions:
            return None, None
        return actions, problem.get_cost_of_actions(actions)

    def next_cell(self, cell, action):
        dx, dy = Actions.direction_to_vector(action)
        return (int(cell[0] + dx), int(cell[1] + dy))

    def visible_invaders(self, game_state):
        """Opponents that are pacmen in our side and that we can see: [(cell, state)]"""
        result = []
        for opp in self.get_opponents(game_state):
            state = game_state.get_agent_state(opp)
            if state.is_pacman and state.get_position() is not None:
                result.append((to_cell(state.get_position()), state))
        return result

    def ghosts(self, game_state):
        """Enemy ghosts that can kill us (cells). The ones we stop seeing are remembered for 2 turns."""
        result = []
        for opp in self.get_opponents(game_state):
            state = game_state.get_agent_state(opp)
            pos = state.get_position()
            if pos is not None and (not state.is_pacman or self.home_dist.get(to_cell(pos), 9) <= 1):
                # (a pacman standing at the border can step back and become a ghost)
                self.ghost_mem[opp] = (to_cell(pos), state.scared_timer, self.turns)
                self.last_ghost_turn = self.turns
            if opp in self.ghost_mem:
                cell, scared, turn = self.ghost_mem[opp]
                if self.turns - turn <= 2 and scared - (self.turns - turn) <= 3:
                    result.append(cell)
        return result

    def update_observations(self, game_state):
        """Food of ours that disappeared since our last turn: the invader that ate it is standing there now."""
        my_pos = to_cell(game_state.get_agent_position(self.index))
        previous = self.get_previous_observation()
        if previous is not None:
            eaten = set(self.get_food_you_are_defending(previous).as_list()) - \
                set(self.get_food_you_are_defending(game_state).as_list())
            if eaten:
                self.invader = (min(eaten, key=lambda f: self.dist(my_pos, f)), self.turns)
                self.last_invader_turn = self.turns
        visible = self.visible_invaders(game_state)
        for cell, _ in visible:
            self.invader = (cell, self.turns)
            self.last_invader_turn = self.turns
        # We should see the estimated position and nobody is there: wrong estimate
        if self.invader and not visible and self.invader[1] < self.turns and \
                util.manhattan_distance(my_pos, self.invader[0]) <= 4:
            self.invader = None

    def choose_action(self, game_state):
        self.turns += 1
        try:
            self.update_observations(game_state)
            action = self.decide(game_state)
            if action in game_state.get_legal_actions(self.index):
                return action
        except Exception:
            pass
        actions = [a for a in game_state.get_legal_actions(self.index) if a != Directions.STOP]
        return random.choice(actions)

    def decide(self, game_state):
        return util.raise_not_defined()

    ########################
    # Offensive behaviour  #
    ########################

    def danger_costs(self, ghosts, near):
        """Cost of every cell of the enemy side: very high next to a ghost, and dead-ends when ghosts are close."""
        cost = {}
        if not ghosts:
            return cost
        for cell in self.walkable:
            if self.is_home(cell):
                continue
            value = 1
            for g in ghosts:
                d = self.dist(g, cell)
                if d <= 1:
                    value += 500
                elif d <= 4:
                    value += (5 - d) * (d == 2 and 12 or 5)
            if near <= 8:
                value += 4 * self.dead_end.get(cell, 0)
            cost[cell] = value
        return cost

    def attack(self, game_state):
        state = game_state.get_agent_state(self.index)
        pos = to_cell(state.get_position())
        carry = self.carried(state)
        home = self.is_home(pos)
        food = self.get_food(game_state).as_list()
        capsules = self.get_capsules(game_state)
        ghosts = self.ghosts(game_state)
        near = min([self.dist(pos, g) for g in ghosts] or [99])
        left = TOTAL_MOVES - self.turns

        # Dynamic decision of going home to deposit the food
        if home or carry == 0:
            self.returning = False
        elif not self.returning:
            limit = 4 if self.turns - self.last_ghost_turn <= 10 else 7
            self.returning = (len(food) <= 3 or left <= self.home_dist[pos] + 4 or near <= 4 or
                              (near <= 7 and carry >= 3) or carry >= limit)

        cost = self.danger_costs(ghosts, near)
        cost_fn = lambda c: cost.get(c, 1)
        enemies_scared = min(game_state.get_agent_state(o).scared_timer for o in self.get_opponents(game_state)) > 8

        if self.returning and not home:
            goals = self.home_entries
        else:
            goals = [f for f in food if self.dist(pos, f) + self.home_dist[f] + 2 <= left]
            if near <= 8 and not enemies_scared:
                goals += capsules  # A capsule scares the ghosts
        actions, total = self.search(pos, goals, cost_fn) if goals else (None, None)
        if actions and total < 400:  # (400 or more means walking next to a ghost)
            return actions[0]
        if self.returning and capsules:
            actions, total = self.search(pos, capsules, cost_fn)
            if actions and total < 400:
                return actions[0]
        if carry > 0 and not home:
            actions, total = self.search(pos, self.home_entries, cost_fn)
            if actions and total < 400:
                return actions[0]
        return self.safe_move(game_state, pos, ghosts)

    def safe_move(self, game_state, pos, ghosts):
        """Nothing safe to do: stay in our side (far from the ghosts) or run away from them."""
        best, best_score = [], -float('inf')
        for action in game_state.get_legal_actions(self.index):
            nxt = self.next_cell(pos, action)
            score = -3.0 if action == Directions.STOP else 0.0
            if self.is_home(nxt):
                score += 40
            else:
                for g in ghosts:
                    d = self.dist(g, nxt)
                    score += -1000 if d <= 1 else 4 * min(d, 8)
                score -= 3 * self.dead_end.get(nxt, 0)
            score -= 0.5 * self.home_dist[nxt]
            if score > best_score:
                best, best_score = [action], score
            elif score == best_score:
                best.append(action)
        return random.choice(best)

    ########################
    # Defensive behaviour  #
    ########################

    def defend(self, game_state):
        state = game_state.get_agent_state(self.index)
        pos = to_cell(state.get_position())
        if not self.is_home(pos):  # We were attacking: go back home first
            cost = self.danger_costs(self.ghosts(game_state), 0)
            actions, _ = self.search(pos, self.home_entries, lambda c: cost.get(c, 1))
            return actions[0] if actions else self.safe_move(game_state, pos, self.ghosts(game_state))

        invaders = self.visible_invaders(game_state)
        if invaders:
            cell, inv = min(invaders, key=lambda i: self.dist(pos, i[0]))
            if state.scared_timer > 0:
                return self.keep_distance(game_state, pos, [i[0] for i in invaders])
            goals = [cell]
            if self.dist(pos, cell) > 2:  # Aim also at its neighbours: the route does not flip if it moves back and forth
                goals += [n for n in self.neighbors[cell] if self.is_home(n)]
            if self.dist(pos, cell) >= 5 and self.carried(inv) >= 2:
                # It carries food and wants to leave: try to cut its way out
                exit_cell = min(self.home_entries, key=lambda e: self.dist(cell, e))
                if self.dist(pos, exit_cell) < self.dist(cell, exit_cell):
                    goals = [exit_cell]
            actions, _ = self.search(pos, goals, home_only=True)
            if actions:
                return actions[0]
        elif self.invader and self.turns - self.invader[1] <= 12 and self.invader[0] != pos:
            # Our food has just disappeared: go there
            actions, _ = self.search(pos, [self.invader[0]], home_only=True)
            if actions:
                return actions[0]
        return self.patrol(game_state, pos)

    def keep_distance(self, game_state, pos, invaders):
        """We are scared (the invader would eat us): stay at distance 2 to keep blocking it."""
        best, best_score = [], -float('inf')
        for action in game_state.get_legal_actions(self.index):
            nxt = self.next_cell(pos, action)
            if not self.is_home(nxt):
                continue
            d = min(self.dist(nxt, i) for i in invaders)
            score = -100.0 + d if d <= 1 else -3.0 * abs(d - 2)
            if score > best_score:
                best, best_score = [action], score
            elif score == best_score:
                best.append(action)
        return random.choice(best or game_state.get_legal_actions(self.index))

    def patrol(self, game_state, pos):
        """Patrol between the two entries closest (on average) to our food."""
        food = self.get_food_you_are_defending(game_state).as_list()
        if not self.patrol_points or self.turns % 15 == 0:
            ranked = sorted(self.home_entries, key=lambda e: sum(self.dist(e, f) for f in food))
            self.patrol_points = []
            for entry in ranked:
                if all(self.dist(entry, p) >= 4 for p in self.patrol_points):
                    self.patrol_points.append(entry)
                if len(self.patrol_points) == 2:
                    break
        target = self.patrol_points[self.patrol_index % len(self.patrol_points)]
        if self.dist(pos, target) <= 1:
            self.patrol_index += 1
            target = self.patrol_points[self.patrol_index % len(self.patrol_points)]
        actions, _ = self.search(pos, [target], home_only=True)
        return actions[0] if actions else Directions.STOP


class OffensiveReflexAgent(ReflexCaptureAgent):
    """Main attacker. It becomes a defender if an invader is close or to protect a lead at the end."""

    def decide(self, game_state):
        state = game_state.get_agent_state(self.index)
        pos = to_cell(state.get_position())
        if self.is_home(pos) and self.carried(state) == 0:
            if self.get_score(game_state) > 0 and TOTAL_MOVES - self.turns <= 40:
                return self.defend(game_state)
            mate_pos = to_cell(game_state.get_agent_position(self.mate(game_state)))
            for cell, _ in self.visible_invaders(game_state):
                if self.dist(pos, cell) <= 4 and self.dist(pos, cell) < self.dist(mate_pos, cell):
                    return self.defend(game_state)
        return self.attack(game_state)


class DefensiveReflexAgent(ReflexCaptureAgent):
    """Defender. It attacks when we are losing (or tied for long) and there are no invaders."""

    def decide(self, game_state):
        state = game_state.get_agent_state(self.index)
        pos = to_cell(state.get_position())
        quiet = not self.visible_invaders(game_state) and self.turns - self.last_invader_turn > 20
        score = self.get_score(game_state)
        scared_home = state.scared_timer > 0 and self.is_home(pos)
        if quiet and not scared_home and TOTAL_MOVES - self.turns >= 25:
            if score < 0 or (score == 0 and self.turns >= 60 and self.turns - self.last_invader_turn > 60):
                return self.attack(game_state)
        return self.defend(game_state) 