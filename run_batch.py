from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import asyncio
from experiment import optimize

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-n", "--novelty", default=None,
                    help="Weight of novelty for averaging. If None: penalizing fitness")
parser.add_argument("-a", "--amount", default=1, type=int, help="Amount of simulations per config.")
parser.add_argument("-s", "--seed", default=1234, type=int, help="Set seed for reproducability.")

args = vars(parser.parse_args())


async def main(novelty_weight: "float|None", amt: int, seed_part:int):
    if novelty_weight is not None:
        novelty_weight = float(novelty_weight)
    for i in range(amt):
        await optimize.main(novelty_weight=novelty_weight, seed_val=int(f"{seed_part}{i}"))

if __name__ == '__main__':
    asyncio.run(main(args["novelty"], args["amount"], args["seed"]))
