import sys
import asyncio
from experiment import optimize


async def main(novelty_weight = None):
    await optimize.main(novelty_weight=novelty_weight)


if __name__ == '__main__':
    novelty_weight = float(sys.argv[1])
    asyncio.run(main(novelty_weight))

