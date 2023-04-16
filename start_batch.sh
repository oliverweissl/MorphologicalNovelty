for run in {1..5}; do
  python3 -m run_batch 0.0 &
  python3 -m run_batch 0.25 &
  python3 -m run_batch 0.5 &
  python3 -m run_batch 0.75 &
  python3 -m run_batch 1.0 &
  python3 -m run_batch &
  sleep 1
done
wait
