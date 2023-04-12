import sys
import asyncio
from experiment import optimize


async def main(novelty_weight = None, amt = 1):
    for _ in range(amt):
    	await optimize.main(novelty_weight=novelty_weight)


if __name__ == '__main__':
    novelty_weight = float(sys.argv[1])
    if len(sys.argv) > 2: 
        amt = int(sys.argv[2])
        asyncio.run(main(novelty_weight, amt))
    else:
        asyncio.run(main(novelty_weight))
