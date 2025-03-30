#/bin/bash
TEST_DIR="./single_test"
# TEST_DIR="./72_test"
LOG_FILE="$TEST_DIR/logs/log-$(date +%Y%m%d%H%M%S).txt"
cp .env.sample .env
# pip install uv
# uv pip install -e '.[test]'
# 不用uv了
pip install -q -e '.[test]'
pytest $TEST_DIR -o log_cli=true > "$LOG_FILE" 2>&1
