cd /home/nel-mout/Documents/VS_Projects/Call_Me_Maybe
uv run python -m src --functions_definition data/input/functions_numbers_only_test.json --input data/input/prompts_numbers_easy.json --output data/output/results_numbers_easy.json
uv run python -m src --functions_definition data/input/functions_numbers_only_test.json --input data/input/prompts_numbers_medium.json --output data/output/results_numbers_medium.json
uv run python -m src --functions_definition data/input/functions_numbers_only_test.json --input data/input/prompts_numbers_hard.json --output data/output/results_numbers_hard.json
uv run python -m src --functions_definition data/input/functions_numbers_only_test.json --input data/input/prompts_numbers_adversarial.json --output data/output/results_numbers_adversarial.json