for run in {1..2}; do
  python3 -m run_batch -n 0.0 -s run &
  python3 -m run_batch -n 0.25 -s run &
  python3 -m run_batch -n 0.5 -s run &
  python3 -m run_batch -n 0.75 -s run &
  python3 -m run_batch -n 1.0 -s run &
  python3 -m run_batch -s run &
  sleep 60
done
wait
