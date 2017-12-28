import random

class SearchTimeout(Exception):
    pass

def custom_score(game, player):
    if game.is_loser(player):
            return float("-inf")
    if game.is_winner(player):
        return float("inf")
    a = len(game.get_legal_moves(player))
    b = len(game.get_legal_moves(game.get_opponent(player)))
    return float(a / (a + b))

def custom_score_2(game, player):
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")
    a = len(game.get_legal_moves(player))
    b = len(game.get_legal_moves(game.get_opponent(player)))
    if not b:
        return float("inf")
    return float(a/b)
 

def custom_score_3(game, player):
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")
    a = len(game.get_legal_moves(player))
    b = len(game.get_legal_moves(game.get_opponent(player)))
    return float(a**2 - b**2)


class IsolationPlayer:
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=20.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

class MinimaxPlayer(IsolationPlayer):
    def get_move(self, game, time_left):
        self.time_left = time_left
        best_move = (-1, -1)
        try:
            return self.minimax(game, self.search_depth)
        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed
        return best_move

    def cut_off_test(self,gameState,depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if depth == 0:
            return True
        if gameState.get_legal_moves():
            return False
        else:
            return True

    def min_value(self,gameState,depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if self.cut_off_test(gameState,depth):
            return self.score(gameState,self)
        legal_moves = gameState.get_legal_moves()
        value = float("inf")
        for move in legal_moves:
            value = min(value, self.max_value(gameState.forecast_move(move),depth-1))
        return value

    def max_value(self,gameState,depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if self.cut_off_test(gameState,depth):
            return self.score(gameState,self)
        legal_moves = gameState.get_legal_moves()
        value = float("-inf")
        for move in legal_moves:
            value = max(value, self.min_value(gameState.forecast_move(move),depth-1))
        return value

    def minimax(self, game, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return (-1,-1)
        best_move = max(legal_moves, key=lambda move: self.min_value(game.forecast_move(move), depth - 1))
        return best_move

class AlphaBetaPlayer(IsolationPlayer):
    def get_move(self, game, time_left):
        self.time_left = time_left
        legal_moves = game.get_legal_moves()
        if legal_moves:
            best_move = legal_moves[0]
        else:
            best_move = (-1, -1)
        depth = 0
        while True:
            try:
                if self.time_left() < self.TIMER_THRESHOLD:
                    raise SearchTimeout()
                depth = depth + 1
                best_move = self.alphabeta(game, depth)
            except SearchTimeout:
                return best_move

    def cut_off_test(self,gameState,depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if depth == 0:
            return True
        legal_moves = gameState.get_legal_moves()
        if legal_moves:
            return False
        else:
            return True

    def min_value(self,gameState,depth,alpha,beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if self.cut_off_test(gameState,depth):
            return self.score(gameState,self)
        legal_moves = gameState.get_legal_moves()
        value = float("inf")
        for move in legal_moves:
            value = min(value, self.max_value(gameState.forecast_move(move),depth-1,alpha,beta))
            if value <= alpha:
                return value
            beta = min(beta,value)
        return value

    def max_value(self,gameState,depth,alpha,beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if self.cut_off_test(gameState,depth):
            return self.score(gameState,self)
        legal_moves = gameState.get_legal_moves()
        value = float("-inf")
        for move in legal_moves:
            value = max(value, self.min_value(gameState.forecast_move(move),depth-1,alpha,beta))
            if value >= beta:
                return value
            alpha = max(alpha,value)
        return value

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return (-1,-1)
        best_move = legal_moves[0]
        best_score = float("-inf")
        for move in legal_moves:
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            value = self.min_value(game.forecast_move(move), depth - 1, alpha, beta)
            if value > best_score:
                best_move = move
                best_score = value
            alpha = max(alpha,best_score)
        return best_move