
#include <stdio.h>

void consolePrint(){
    printf("hello world\n");
}

int addOne(int num){
    return num +1;
}

int someFunc(int num_numbers, int *numbers){
    int i;
    int s = 0;

    for (i = 0; i < num_numbers; i++)
    {
        s += numbers[i];
    }

    return s;
    
}