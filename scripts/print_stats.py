import pstats
import sys

dump = sys.argv[1]
top_k = int(sys.argv[2])

p = pstats.Stats(dump).strip_dirs().sort_stats('time')
p.print_stats(top_k)
