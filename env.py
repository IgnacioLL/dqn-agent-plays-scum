import collections
import random
from typing import List, Tuple

import numpy as np

from constants import Constants as C

class ScumEnv:
    def __init__(self, number_players: int):
        self.number_players = number_players
        self.cards = self._deal_cards()
        self.player_turn = 0
        self.last_player = -1
        self.last_move = None
        self.players_in_game = [True] * self.number_players
        self.players_in_round = self.players_in_game.copy()
        self.n_players_in_round = sum(self.players_in_round)
        self.player_position_ending = 0
        self.player_order = [-1] * self.number_players

    def _get_deck(self) -> List[int]:
        deck = [rank for _ in range(C.NUMBER_OF_SUITS) for rank in range(1, C.NUMBER_OF_CARDS_PER_SUIT + 1)]
        deck.remove(C.NUMBER_OF_CARDS_PER_SUIT)
        deck.append(C.NUMBER_OF_CARDS_PER_SUIT + 1)  # 2 of hearts
        return deck

    def _deal_cards(self) -> List[List[List[int]]]:
        deck = self._get_deck()
        sampled_data = random.sample(deck, C.NUMBER_OF_CARDS)
        
        cards_per_player = C.NUMBER_OF_CARDS // self.number_players
        remainder = C.NUMBER_OF_CARDS % self.number_players
        
        cards_of_players = []
        for i in range(self.number_players):
            start = i * cards_per_player + min(i, remainder)
            end = start + cards_per_player + (1 if i < remainder else 0)
            player_cards = sorted(sampled_data[start:end])
            cards_of_players.append(self._get_combinations(player_cards))

        return cards_of_players

    @staticmethod
    def _extract_higher_order(cards: List[int], number_of_repetitions: int) -> List[int]:
        return list(set(item for item, count in collections.Counter(cards).items() if count >= number_of_repetitions))

    @classmethod
    def _get_combinations(cls, cards: List[int]) -> List[List[int]]:
        return [
            cards,
            cls._extract_higher_order(cards, 2),
            cls._extract_higher_order(cards, 3),
            cls._extract_higher_order(cards, 4)
        ]

    def reset(self) -> None:
        self.__init__(self.number_players)

    def _next_turn(self) -> int:
        next_player = (self.player_turn + 1) % self.number_players
        while not self.players_in_round[next_player]:
            next_player = (next_player + 1) % self.number_players
            if next_player == self.player_turn:
                break
        return next_player

    def _update_player_turn(self, skip: bool = False) -> None:
        if skip: print("Skip!!")
        turns_to_skip = 2 if skip else 1
        for _ in range(turns_to_skip):
            self.player_turn = self._next_turn()

    def _update_player_cards(self, card_number: int, n_cards: int) -> None:
        if card_number == C.NUMBER_OF_CARDS_PER_SUIT + 1:  # two of hearts
            self.cards[self.player_turn][0].remove(card_number)
        else:
            for _ in range(n_cards + 1):
                self.cards[self.player_turn][0].remove(card_number)
        self.cards[self.player_turn] = self._get_combinations(self.cards[self.player_turn][0])

    def _reinitialize_round(self) -> None:
        print("End of round\n" + "_" * 40 + "\n")
        self._print_players_in_game()
        self.last_player = -1
        self.last_move = None
        self.players_in_round = self.players_in_game.copy()
        self.n_players_in_round = sum(self.players_in_round)

    def _check_players_playing(self) -> None:
        if self.n_players_in_round <= 1:
            self.player_turn = self.last_player
            self._reinitialize_round()

    def _check_player_finish(self) -> bool:
        if not self.cards[self.player_turn][0]:
            print(f" -> Player {self.player_turn} finished!!")
            self.players_in_game[self.player_turn] = False
            self.player_order[self.player_position_ending] = self.player_turn
            self.player_position_ending += 1
            self.last_move = None
            self._reinitialize_round()
            return True
        return False

    def _get_cards_to_play(self) -> Tuple[List[int], int]:
        if self.last_move is None:
            return self._get_cards_to_play_init()
        else:
            return self._get_cards_to_play_followup()

    def _get_cards_to_play_init(self) -> Tuple[List[int], int]:
        available_combinations = [i for i in range(C.NUMBER_OF_SUITS) if self.cards[self.player_turn][i]]
        n_cards = random.choice(available_combinations)
        return self.cards[self.player_turn][n_cards], n_cards

    def _get_cards_to_play_followup(self) -> Tuple[List[int], int]:
        n_cards = self.last_move[1]
        two_of_hearts = [C.NUMBER_OF_CARDS_PER_SUIT + 1] if C.NUMBER_OF_CARDS_PER_SUIT + 1 in self.cards[self.player_turn][0] else []
        possibilities = self.cards[self.player_turn][n_cards]

        if self.last_move[0] == 5:
            return [cards for cards in possibilities if cards in [5, 6]], n_cards
        elif self.last_move[0] == 8:
            return [cards for cards in possibilities if cards <= self.last_move[0]], n_cards
        else:
            return [cards for cards in possibilities if cards >= self.last_move[0]] + two_of_hearts, n_cards

    def make_move(self) -> None:
        if sum(self.players_in_game) == 0:
            print("_" * 24 + "\nEnd of game\n" + "_" * 24)
            return

        self._print_game_state()

        if self.last_player == self.player_turn:
            self._reinitialize_round()
            return

        cards_to_play_with, n_cards = self._get_cards_to_play()

        if not cards_to_play_with:
            self._handle_unable_to_play()
            return

        card_number = random.choice(cards_to_play_with)
        print(f"The player moved {C.N_CARDS_TO_TEXT[n_cards+1]} {str(card_number)}")

        skip = self.last_move is not None and self.last_move[0] == card_number

        self._update_player_cards(card_number, n_cards)
        self.last_move = [card_number, n_cards]
        finish = self._check_player_finish()

        if finish:
            skip = False

        self.last_player = self.player_turn
        self._update_player_turn(skip=skip)

    def _print_game_state(self) -> None:
        if self.last_move is not None:
            print(f"Last move was: {C.N_CARDS_TO_TEXT[self.last_move[1]]} {str(self.last_move[0])}, played by {self.last_player}")
        else:
            print("Beginning of round")
        print("The player who has to move is: ", self.player_turn)
    
    def _print_players_in_game(self) -> None:
        players_playing = ", ".join([str(i) for i, x in enumerate(self.players_in_game) if x])
        print(f"Players playing are: {players_playing}")

    def _handle_unable_to_play(self) -> None:
        print(f"Player number {self.player_turn} is unable to play")
        self.players_in_round[self.player_turn] = False
        self.n_players_in_round -= 1
        self._update_player_turn()
        self._check_players_playing()

if __name__ == '__main__':
    env = ScrumEnv(5)
    for _ in range(100):
        env.make_move()