import sys
import asyncio
from experiment import optimize


async def main(novelty_search, novelty_weight = None):
    await optimize.main(novelty_search=novelty_search, novelty_weight=novelty_weight)


if __name__ == '__main__':
    if len(sys.argv) > 2:
        novelty_search, novelty_weight = bool(sys.argv[1]), float(sys.argv[2])
        asyncio.run(main(novelty_search, novelty_weight))
    else:
        novelty_search = bool(sys.argv[1])
        asyncio.run(main(novelty_search))

