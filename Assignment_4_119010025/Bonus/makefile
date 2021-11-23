CC=nvcc
RM=rm
main.out: main.cu file_system.cu user_program.cu
	$(CC) -rdc=true main.cu file_system.cu user_program.cu -o main.out
clean:
	$(RM) main.out

