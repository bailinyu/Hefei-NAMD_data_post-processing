ls -l {0001..0500}/WAVECAR | awk '{print $5}' | uniq -c
