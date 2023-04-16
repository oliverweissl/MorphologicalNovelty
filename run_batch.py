import sys
import asyncio
from experiment import optimize


async def main(novelty_weight: "float|None" = None, amt: int = 1):
    for _ in range(amt):
        await optimize.main(novelty_weight=novelty_weight)

if __name__ == '__main__':
    if len(sys.argv) > 2:
        asyncio.run(main(float(sys.argv[1]), int(sys.argv[2])))
    else:
        if len(sys.argv) > 1:
            asyncio.run(main(amt=int(sys.argv[1])))
        else:
            asyncio.run(main())
