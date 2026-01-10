import random
import numpy as np
import statistics

def sum10(x):
    count = 0
    pair = []
    for i in range(len(x)):
        ind = i
        for j in range(len(x)):
            if (x[i]+x[j]) == 10:
                if (i == j):    
                    break
                if x[j] in pair:
                    break
                else:
                    count+=1
                    pair.append(x[i])
                    pair.append(x[j])
    return count

def rangmaxmin(x):
    max = x[0]
    min = x[0]
    for i in range(len(x)):
        if max < x[i]:
            max = x[i]
        if min > x[i]:
            min = x[i]
    return f"Range : {max-min}"

def Exc(x):
    if len(x)<3:
        raise ValueError("Range determination not possible")
    return rangmaxmin(x)

def highchar(x):
    ch = ""
    count = 0
    for i in x:
        if count < x.count(i):
            count = x.count(i)
            ch = i
    return ch

def highcharcount(x):
    ch = ""
    count = 0
    for i in x:
        if count < x.count(i):
            count = x.count(i)
            ch = i
    return count

def mean(x):
    li = np.array(x)
    return np.mean(li)

def median(x):
    li = np.array(x)
    return np.median(li)

def mode(x):
    return statistics.mode(x)

def matmult(x,y):
    return np.dot(x,y)

def matpow(a,m):
    res = matmult(a,a)
    for i in range(m-2):
        res = matmult(res,a)
    return res

    
def main():
    print("1.Pairs of elements with sum equal to 10")
    print("2.Range of the list")
    print("3.Matrix Power")
    print("4.Highest occurring character")
    print("5.Mean , Median , Mode ")
    ch = int(input("Enter choice : "))

    if ch == 1:
        n1 = int(input("Enter number of elements to be added in the list : "))
        list1 = []
        for i in range(n1):
            e = int(input(f"Enter element {i+1} : "))
            list1.append(e)
        print(f"Pairs of elements with sum equal to 10 : {sum10(list1)}")

    elif ch == 2:
        n2 = int(input("Enter number of elements to be added in the list : "))
        list2 = []
        for i in range(n2):
            e = int(input(f"Enter element {i+1} : "))
            list2.append(e)
        try :
            print(Exc(list2))
        except ValueError as e :
            print(e)

    elif ch == 3:
        x = int(input("Enter dimension of square matrix : "))
        A = []
        for i in range(x):
            row = []
            for j in range(x):
                e = int(input(f"Enter element [{i+1}][{j+1}] : "))
                row.append(e)
            A.append(row)
        m = int(input("Enter power of matrix : "))
        print(f"Result of A^m is : {matpow(A,m)}")

    elif ch == 4:
        word = input("Enter word : ")
        print(f"highest occurring character is {highchar(word)} and its count is {highcharcount(word)}")

    elif ch == 5:
        list5 = []
        for i in range(25):
            x = random.randint(1, 10)
            list5.append(x)
        print(list5)
        print(f"Mean : {mean(list5)}")
        print(f"Median : {median(list5)}")
        print(f"Mode : {mode(list5)}")

    else:
        print("Invalid choice")

main()


