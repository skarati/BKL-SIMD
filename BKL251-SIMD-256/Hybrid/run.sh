clear
gcc -o main main.c -march=native -mtune=native -mavx512f -m64 -O3 -funroll-loops -fomit-frame-pointer -no-pie 
./main
