# export OPENAI_API_KEY=
# export LMUData="./LMUData"
# export SiliconFlow_API_KEY=


# valid_type: STR, LLM
python -u run.py --data PhyX_mini \
    --model InternVL2_5-8B \
    --judge deepseek --judge-args '{"valid_type": "STR"}'