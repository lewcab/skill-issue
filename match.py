"""
Match Class for Esports Data
"""


class Match:
    """
    Represents a match in the esports data, keeping track of features used for training a model.
    This class can be extended to include more match-related attributes and methods.
    Attributes are prefixed with `b_` and `r_` to denote blue and red side teams respectively.
    The number following the prefix indicates the player role with `1` for top, `2` for jungle, etc.
    If no number is present, it refers to the team as a whole.
    Player statistics are an aggregate of the past 20 (or fewer) official matches.
    """
    def __init__(
            self,
            winner: bool,
            b_fb: float,
            b_wr: float,
            b_1_gpm: float,
            b_1_kda: float,
            b_2_gpm: float,
            b_2_kda: float,
            b_3_gpm: float,
            b_3_kda: float,
            b_4_gpm: float,
            b_4_kda: float,
            b_5_gpm: float,
            b_5_kda: float,
            r_fb: float,
            r_wr: float,
            r_1_gpm: float,
            r_1_kda: float,
            r_2_gpm: float,
            r_2_kda: float,
            r_3_gpm: float,
            r_3_kda: float,
            r_4_gpm: float,
            r_4_kda: float,
            r_5_gpm: float,
            r_5_kda: float,
):
        self.winner = winner
        self.b_fb = b_fb
        self.b_wr = b_wr
        self.b_1_gpm = b_1_gpm
        self.b_1_kda = b_1_kda
        self.b_2_gpm = b_2_gpm
        self.b_2_kda = b_2_kda
        self.b_3_gpm = b_3_gpm
        self.b_3_kda = b_3_kda
        self.b_4_gpm = b_4_gpm
        self.b_4_kda = b_4_kda
        self.b_5_gpm = b_5_gpm
        self.b_5_kda = b_5_kda
        self.r_fb = r_fb
        self.r_wr = r_wr
        self.r_1_gpm = r_1_gpm
        self.r_1_kda = r_1_kda
        self.r_2_gpm = r_2_gpm
        self.r_2_kda = r_2_kda
        self.r_3_gpm = r_3_gpm
        self.r_3_kda = r_3_kda
        self.r_4_gpm = r_4_gpm
        self.r_4_kda = r_4_kda
        self.r_5_gpm = r_5_gpm
        self.r_5_kda = r_5_kda

    def __repr__(self):
        return (
            f"Match(winner={self.winner}\n"
            f"b_fb={self.b_fb}, b_wr={self.b_wr}, "
            f"b_1_gpm={self.b_1_gpm}, b_1_kda={self.b_1_kda}, "
            f"b_2_gpm={self.b_2_gpm}, b_2_kda={self.b_2_kda}, "
            f"b_3_gpm={self.b_3_gpm}, b_3_kda={self.b_3_kda}, "
            f"b_4_gpm={self.b_4_gpm}, b_4_kda={self.b_4_kda}, "
            f"b_5_gpm={self.b_5_gpm}, b_5_kda={self.b_5_kda},\n"
            f"r_fb={self.r_fb}, r_wr={self.r_wr}, "
            f"r_1_gpm={self.r_1_gpm}, r_1_kda={self.r_1_kda}, "
            f"r_2_gpm={self.r_2_gpm}, r_2_kda={self.r_2_kda}, "
            f"r_3_gpm={self.r_3_gpm}, r_3_kda={self.r_3_kda}, "
            f"r_4_gpm={self.r_4_gpm}, r_4_kda={self.r_4_kda}, "
            f"r_5_gpm={self.r_5_gpm}, r_5_kda={self.r_5_kda})"
        )

def main():
    print("Creating a Match instance...")
    match = Match(
        winner=True,
        b_fb=0.5,
        b_wr=0.6,
        b_1_gpm=300,
        b_1_kda=2.5,
        b_2_gpm=320,
        b_2_kda=3.0,
        b_3_gpm=310,
        b_3_kda=2.8,
        b_4_gpm=290,
        b_4_kda=2.0,
        b_5_gpm=280,
        b_5_kda=1.5,
        r_fb=0.4,
        r_wr=0.5,
        r_1_gpm=310,
        r_1_kda=2.2,
        r_2_gpm=300,
        r_2_kda=2.5,
        r_3_gpm=290,
        r_3_kda=2.0,
        r_4_gpm=280,
        r_4_kda=1.8,
        r_5_gpm=270,
        r_5_kda=1.2
    )
    print(match)


if __name__ == '__main__':
    main()
