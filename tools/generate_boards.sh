#!/bin/bash
# Generate board pools using TypeScript CLI

set -e

echo "Generating board pools for training..."
echo ""

cd "$(dirname "$0")/../../spaces-game-engine"

# Size 2 - Complete exhaustive (~16 boards)
echo "Generating size 2 boards..."
npm run cli -- generate-boards --size 2 --limit 500 --output ../spaces-game-python/data/boards_size_2.json

# Size 3 - Training baseline (~500 boards)
echo "Generating size 3 boards..."
npm run cli -- generate-boards --size 3 --limit 500 --output ../spaces-game-python/data/boards_size_3.json

# Size 4 - Intermediate (~5000 boards)
echo "Generating size 4 boards..."
npm run cli -- generate-boards --size 4 --limit 5000 --output ../spaces-game-python/data/boards_size_4.json

# Size 5 - Advanced (~50000 boards)
echo "Generating size 5 boards..."
npm run cli -- generate-boards --size 5 --limit 50000 --output ../spaces-game-python/data/boards_size_5.json

echo ""
echo "✅ Board generation complete!"
echo ""
echo "Generated files:"
ls -lh ../spaces-game-python/data/boards_size_*.json
