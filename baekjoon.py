"""
#상수 거꾸로 비교  2908  
a, b=input().split()
a=int(a[::-1])
b=int(b[::-1])

if a>b:
    print(a)
else:
    print(b)


#나머지 구하기 3052

a=[]
for i in range (10):
    b=int(input())
    a.append(b%42)
    
    
print(len(set(a)))


#최댓값 최솟값 구하기 10818
n = int(input())
a=list(map(int, input().split()))

print(min(a), max(a))    


# 숫자의 합
n=int(input())
num=list(map(int,input()))
total = 0
for i in range(n):
    total += int(num[i])
print(total)


#최댓값
mn=[]
for i in range(9):
    mn.append(int(input()))
    
print(max(mn))
print(mn.index(max(mn))+1)


# OX문제
n = int(input())

for _ in range(n):
    ox_list = list(input())
    score = 0  
    sum_score = 0  # 새로운 ox리스트를 입력 받으면 점수 합계를 리셋한다.
    for ox in ox_list:
        if ox == 'O':
            score += 1  # 'O'가 연속되면 점수가 1점씩 커진다.
            sum_score += score  # sum_score = sum_score + score
        else:
            score = 0
    print(sum_score)

#OX 문제 8958
n=int(input())

for i in range (n):
    ox=list(map(str,input()))
    cnt=0
    cnt_list=0
    for i in range (len(ox)):
        if ox[i]=='O':
            cnt+=1
            cnt_list+=cnt
        else:
            cnt=0
    print(cnt_list)


# 음계 2920
a = list(map(int, input().split()))
 
if a == sorted(a):
    print('ascending')
elif a == sorted(a, reverse=True):
    print('descending')
else:
    print('mixed')


#체스판 다시 칠하기 1018
col,row=map(int,input().split())
result=[]
board=[]

for i in range(col):
    board.append(input())

for i in range(col-7):
    for j in range(row-7):
        draw1=0
        draw2=0
        for a in range(i, i+8):
            for b in range(j, j+8):
                if(a+b)%2==0:
                    if board[a][b]!="B":
                        draw1 += 1
                    if board[a][b]!="W":
                        draw2 += 1
    
                else:
                    if board[a][b]!="W":
                        draw1 += 1
                    if board[a][b]!="B":
                        draw2 += 1
                        
        result.append(draw1)
        result.append(draw2)
        
        
print(min(result))


#직사각형 탈출 1085

a,b,c,d=map(int,input().split())
sibal =0
jot = 0
if a<int(c-a):
    sibal=a
else:
    sibal=(c-a)
    
if (b)<(d-b):
    jot=b
else:
    jot=d-b

if sibal>jot:
    print(jot)
else:
    print(sibal)



# 단어 정렬 1181
n = int(input())
n_list=[]
for i in range (n):
    n_list.append(str(input()))

n_list=set(n_list)
n_list=list(n_list)
n_list.sort()
n_list.sort(key=len)


for i in n_list:
    print(i)


#팰린드롬수 1259 
n=1
while n!='0':
    n = input()
        
    if n==n[::-1]:
        print("yes")
    else:
        print("no")
 


#영화감독 숌 1436
n=int(input())
str_6=666
cnt=0
while True:
    if '666' in str(str_6):
        cnt+=1
    if cnt== n:
        print(str_6)
        break
    str_6 +=1



#스택 수열 1874
n = int(input())
n_list=[]
op=[]

for i in range (n):
    num=int(input())
    n_list.append(num)
    op.append('+')
    
    if n_list[-1]==



count = 1
temp = True
stack = []
op = []

N = int(input())
for i in range(N):
    num = int(input())
    # num이하 숫자까지 스택에 넣기
    while count <= num:
        stack.append(count)
        op.append('+')
        count += 1

    # num이랑 스택 맨 위 숫자가 동일하다면 제거
    if stack[-1] == num:
        stack.pop()
        op.append('-')
    # 스택 수열을 만들 수 없으므로 NO
    else:
        temp = False
        break

# 스택 수열을 만들수 있는지 여부에 따라 출력 
if temp == False:
    print("NO")
else:
    for i in op:
        print(i)





#수 찾기 1920
n = int(input())
n_list=set(map(int, input().split()))

m= int (input())
m_list=list(map(int,input().split()))
    
cnt=0
for i in m_list:
    if i in n_list:
        print(1)
    else:
        print(0)




#m이상 n이하 소수  출력하기  1929
a,b=map(int,input().split())
for i in range(a,b+1):
    if i==1:
        continue
    for j in range(2,int(i**0.5)+1):
        if i % j ==0:
            break
    else:
        print(i)



#소수 찾기 1978
n = int(input())
n_list=map(int, input().split())
cnt=0
for i in n_list:
    error=0
    if i>1:
        for j in range (2, i): # 2부터 n-1까지
            if i%j==0:
                error +=1 # 2부터 n-1까지 나눈 몫이 0이면 error가 증가
        if error==0:
            cnt+=1 # error가 없으면 소수.
print(cnt)



#통계학 2108
import math

n=int(input())
n_list=[]
for i in range(n):
    n_list.append(input())
    
print(avg(n_list))
print(n_list.index(n/2))

# 통계학 2108
import numpy
n=int(input())
a_list=[]
for i in range(n):
    a=int(input())
    a_list.append(a)
res=sum(a_list)
mid=n//2

sort_list=sorted(a_list)

#최빈값
max_input=max(a_list)
my_input=[0]*max_input
for i in a_list:
    my_input[i-1]+=1    
max_list=max(my_input)
#범위
min_input=min(a_list)
print(round(res/n))
print(sort_list[mid])
print(max_list)
print(max_input-min_input)


#2164 카드2
from collections import deque

N = int(input())
deque = deque([i for i in range(1, N+1)])

while(len(deque) >1):
    deque.popleft()
    move_num = deque.popleft()
    deque.append(move_num)
    
print(deque[0])


#2231 분해합
N = int(input())
result = 0

for i in range(1, N + 1):
    tmp = i + sum(map(int,str(i)))

    if tmp == N:
        result = i
        break

print(result)



# 벌집 2292
n=int(input())

cnt=1
turn=1
while n >turn:
    turn += 6*cnt
    cnt+=1
print(cnt)


#2775
n=int(input())

for i in range(n):
    floor=int(input())
    room=int(input())

    zero_floor=[x for x in range(1, room+1)]
    
    for j in range(floor):  # 2 32
        for e in range(1,room):
            zero_floor[e]+=zero_floor[e-1]
    print(zero_floor[-1])



#2798
import itertools

N, M = map(int, input().split())
card_num = list(map(int, input().split()))

# M을 넘지 않으면서 M에 최대한 가까운 카드 3장의 합
combi_sum = [sum(combi) for combi in itertools.combinations(card_num, 3) if sum(combi) <= M]

print(max(combi_sum))



#2805 나무 자르기

n, wood=map(int,input().split())
n_list=list(map(int,input().split())) 
res=[]
ans=0
for i in n_list:
    if i-wood >= wood :
        res.append(i)  
    else:
        continue
ans=sum(res)-wood    

print(ans//(len(res)))


import sys
input = sys.stdin.readline

N, M = map(int,input().split()) # 나무 수, 필요한 나무 길이
trees = list(map(int, input().split()))

start, end = 0, max(trees) # 시작 점, 끝점

# 이분 탐색
while start <= end:
    mid = (start+end)//2
    tree = 0 # 잘린 나무 합
    for i in trees:
        if i > mid: # mid보다 큰 나무 높이는 잘림
            tree += i - mid

    if tree >= M: # 원하는 나무 높이보다 더 많이 잘렸으면
        start = mid + 1
    else: # 원하는 나무 높이보다 덜 잘렸으면
        end = mid - 1
print(end)
"""

"""
직각삼각형 
    lst = list(map(int, input().split()))
    if lst[0] == 0 and lst[1] == 0 and lst[2] == 0:
        break
    lst.sort()
    if lst[2]**2 == lst[0]**2 + lst[1]**2:  # 피타고라스 정리 활용
        print('right')
    else:
        print('wrong')
"""
"""
#나이순정렬
n = int(input())
member_lst = []

for i in range(n):
    age, name = map(str, input().split())
    age = int(age)
    member_lst.append((age, name))

member_lst.sort(key = lambda x : x[0])	## (age, name)에서 age만 비교

for i in member_lst:
    print(i[0], i[1])
 """
"""   
 # 제로 10773
n = int (input())
nlist=[]
res=0
for i in range(n):
    num=int(input())
    if num==0:
        nlist.pop()
    else:
        nlist.append(num)

res=sum(nlist)
print(res)
"""
"""
#2839 설탕나누기
n=int(input())
cnt=0
while n>=0:
    if n%5==0:
        cnt = cnt+( n//5 )
        print(cnt)
        break
    cnt -=3
    cnt+=1
else:
    print(-1) 
"""
"""
#11050 이항계수 구하기

from math import prod
n,k = map(int,input().split())

boonmo=[]
boonmo2=[]
boonja=[]
for j in range(1,n+1):
    boonja.append(j)

for i in range(1,n-k+1):
    boonmo.append(i)

for w in range(1,k+1):
    boonmo2.append(w)

a=prod(boonja)
b=prod(boonmo)
c=prod(boonmo2)

res=a/(b*c)
print(res)

from math import factorial
n, k = map(int, input().split())
b = factorial(n) // (factorial(k) * factorial(n - k))
print(b)
"""
"""
수 정렬하기 10989
n=int(input())
nlist=[0]*10001
for i in range(n):
    num=int(input())
    nlist[num-1] +=1

for i in range(10000):
    if nlist[i] !=0:
        for j in range(nlist[i]):
            print(i+1)
"""
"""
#숫자카드비교 10816
import sys
input = sys.stdin.readline

dict={}
n=int(input())
nlist=list(map(int,input().split()))
k=int(input())
klist=list(map(int,input().split()))
res=[]
for j in range(n):
    if nlist[j] in dict:
        dict[nlist[j]] +=1
    else:
        dict[nlist[j]] =1

for i in range(k):
    if klist[i] in dict:
        print(dict[klist[i]],end=' ')
    else:
        print(0,end_=_'')

"""
"""
#좌표 정렬 11650
n = int(input())
nlist=[]
for i in range(n):
    a,b = map(int,input().split())
    nlist.append((a,b))

nlist.sort()

for x,y in nlist:
    print(x,y)
"""
"""
#좌표 정렬 11651
import sys
n = int(sys.stdin.readline().split())
nlist=[]
for i in range(n):
    a,b = map(int,sys.stdin.readline().split())
    nlist.append((a,b))

alist=sorted(nlist,key=lambda x: (x[1],x[0]))

for x,y in alist:
    print(x,y)
"""

"""
11866 
a,b = map(int,input().split())
nlist=list(range(1,a+1))
c=0 
newlist=[]   
for j in range(1,a+1):
    if a> b*j:  # 3 6 2 7 5 1 4
        c=c+b
        newlist.append(nlist.pop(c-1))   # 3  6
    else:   # 1 2 4 5 7
        c=b*j-a
        c=c+b
        newlist.append(nlist.pop(-1))


print(newlist)

""""""
from collections import deque

k,n = map(int, input().split())
kdeque = deque([i for i in range(1, k+1)])

print("<",end='')
while kdeque:
    for i in range(n):
        kdeque.append(kdeque[0])
        kdeque.popleft()
    print(kdeque.popleft(),end='')
    if kdeque:
        print(",",end=' ')
print(">")
        
"""
"""
#10866 덱
from collections import deque

n=int(input())
s=deque()
for i in range(n):
    order=list(input().split())
    if order[0]=="push_front":
        s.appendleft(order[1])
    elif order[0]=="push_back":
        s.append(order[1])
    elif order[0]=="pop_front":
        if s:
            print(s.popleft())
        else:
            print(-1)
    elif order[0]=="pop_back":
        if s:
            print(s.pop())
        else:
            print(-1)
    elif order[0]=="size":
        print(len(s))
    elif order[0]=="empty":
        if s:
            print(0)
        else:
            print(1)
    elif order[0]=="front":
        if s:
            print(s[0])
        else:
            print(-1)
    elif order[0]=="back":
        if s:
            print(s[-1])
        else:
            print(-1)

"""
"""
import sys

n=int(sys.stdin.readline())
res=[]
for i in range(n):
    s=list(sys.stdin.readline().split())
    if s[0]=="push":
        res.append(s[1])
    elif s[0]=="pop":
        if res:
            print(res.pop())
        else:
            print(-1)
    elif s[0]=="size":
        print(len(res))
    elif s[0]=="empty":
        if res:
            print(0)
        else:
            print(-1)
    elif s[0]=="top":
        if res:
            print(res[-1])
        else:
            print(-1)
        """
"""
#10845 큐
import sys

n= int(sys.stdin.readline())
res=[]
for i in range(n):
    s=list(sys.stdin.readline().split())
    if s[0]=="push":
        res.append(s[1])
    elif s[0]=="pop":
        if res:
            print(res.pop(0))
        else:
            print(-1)
    elif s[0]=="size":
        print(len(res))
    elif s[0]=="empty":
        if res:
            print(0)
        else:
            print(1)
    elif s[0]=="front":
        if res:
            print(res[0])
        else:
            print(-1)
    elif s[0]=="back":
        if res:
            print(res[-1])
        else:
            print(-1)
"""
"""
#7568 덩치비교

N = int(input())
people = [list(map(int,input().split())) for _ in range(N)]
result = [1]*N

# 덩치 비교 (덩치가 작은 사람 등수 +1)
for i in range(N-1):
    for j in range(i+1,N):
    
        # i번째 사람이 덩치가 더 큰가?
        if people[i][0] > people[j][0] and people[i][1] > people[j][1]:
            result[j] += 1
            
        # j번째 사람이 덩치가 더 큰가?
        elif people[i][0] < people[j][0] and people[i][1] < people[j][1]:
            result[i] += 1
            
print(*result)
"""
"""
#9012 괄호
n=int(input())
for i in range(n):
    k=list(input())
    hap=0
    for i in k:
        if i=="(":
            hap+=1
        else:
            hap-=1
        if hap < 0 :
            print("NO")
            break
    if hap==0:
        print("YES")
    elif hap >0:
        print("NO")
"""
"""
#4949 균형잡힌 세상

def check(n):
    s=[]
    for i in n:
        if i=="(":
            s.append(i)
        elif i==")":
            if not s or s[-1]!="(":
                return 'no'
            else:
                s.pop()
        elif i=="[":
            s.append(i)
        elif i=="]":
            if not s or s[-1]!="[":
                return 'no'
            else:
                s.pop()
        
    if s:
        return 'no'
    return 'yes'


while True:
    n=input()
    if n==".":
        break
    print(check(n))
"""
"""
#1654 랜선자르기
import sys

n,k = map(int,sys.stdin.readline().split())
a=[]
for i in range(n):
    a.append(int(input()))

start=1
end=max(a)
while start<=end:
    mid=(start+end)//2
    cnt=0
    for i in a:
        cnt+= i //mid
    if cnt>=k:
        start=mid+1
    else:
        end=mid-1
print(end)
"""
"""
#10250 ACM 호텔

n=int(input())
for i in range(n):
    h,w,c=map(int,input().split())
    res=(c%h)*100+(c//h)+1
    print(res)
"""
"""
#18111 마인크래프트
import sys

n, m, b = map(int, sys.stdin.readline().split())
graph = [list(map(int, sys.stdin.readline().split())) for _ in range(n)]
answer = sys.maxsize
idx = 0

# 0층부터 256층까지 반복
for target in range(257):
    max_target, min_target = 0, 0

    # 반복문을 통해 블록을 확인
    for i in range(n):
        for j in range(m):

            # 블록이 층수보다 더 크면
            if graph[i][j] >= target:
                max_target += graph[i][j] - target

            # 블록이 층수보다 더 작으면
            else:
                min_target += target - graph[i][j]

    # 블록을 뺀 것과 원래 있던 블록의 합과 블록을 더한 값을 비교
    # 블록을 뺀 것과 원래 있던 블록의 합이 더 커야 층을 만들 수 있음.
    if max_target + b >= min_target:
        # 시간 초를 구하고 최저 시간과 비교 
        if min_target + (max_target * 2) <= answer:
        	# 0부터 256층까지 비교하므로 업데이트 될수록 고층의 최저시간
            answer = min_target + (max_target * 2) # 최저 시간
            idx = target # 층수

print(answer, idx)
---------------------------------------------------------------------

n,m,b = map(int,input().split()) 
mat=[list(map(int,input().split())) for _ in range(n)]
lo=min([min(x) for x in mat])
hi=max([max(x) for x in mat])
INF=int(1e9)

def check(mat,lv,b):
    sc=0
    c=0
    for i in range(n):
        for j in range(m):
            z=mat[i][j]-lv
            if z>0:
                b+=z
                sc+=2*z
            else:
                c+=-z
    if b<c:
        return INF
    return sc+c

msc=INF
mlv=0
for lv in range(hi,lo-1,-1):
    sc=check(mat,lv,b)
    if msc>sc:
        msc=sc
        mlv=lv
print(msc,mlv)
"""
"""
#15829 hashing

L = int(input())
string = input()
answer = 0

for i in range(L):
    answer += (ord(string[i])-96) * (31 ** i) #아스키 코드 값을 돌려주는 ord함수
print(answer % 1234567891)
"""
"""
#`1003피보나치 함수
1 1 2 3 5 8
1 1 2 3
f(0) f(1) f(2)     f(3)  f(4)       f(5)                 f(6)
0    1     0,1    0,1,1   0,0,1,1,1  0,0,0,1,1,1,1,1   0,0,0,0,0,1,1,1,1,1,1,1,1

def fibo(num):
    zero=[1,0,1]
    one=[0,1,1]
    lenght=len(zero)
    if num>=lenght:
        for i in range(lenght,num+1):
            zero.append(zero[i-2]+zero[i-1])
            one.append(one[i-2]+one[i-1])
    print('{} {}'.format(zero[num],one[num]))


n=int(input())
for i in range(n):
    fibo(int(input()))
"""
"""
#11723 집합 문제
import sys

n= int(sys.stdin.readline())
res=set()
for i in range(n):
    s=sys.stdin.readline().split()
    if len(s)==1:
        if s[0]=="all":
            res = set([i for i in range(1, 21)])
        else:
            res=set()
    else:
        aa,bb=s[0],s[1]
        bb=int(bb)
        if aa=="add":
            res.add(bb)
        elif aa=="remove":
            res.discard(bb)    
        elif aa=="check":
            print( 1 if bb in res else 0)
        elif aa=="toggle":
            if bb in res:
                res.discard(bb)
            else:
                res.add(bb)
"""
"""
#1676 팩토리얼 0의 개수
import math as mt

n=int(input())
res=mt.factorial(n)
result=list(map(int,str(res)))
result=result[::-1]
cnt=0
for i in result:
    if i!=0:
        break
    else:
        cnt+=1

print(cnt)
"""
#1５４１팩토리얼 0의 개수


s=input().split('-')
res=[]
for i in s:
    cnt=0
    k=i.split("+")
    for j in k:
        cnt+=int(j)
        res.append(cnt)
n=res[0]
for i in range(1,len(res)):
    n-=res[i]

print(n)
a = input().split('-')
num = []
for i in a:
    cnt = 0
    s = i.split('+')
    for j in s:
        cnt += int(j)
    num.append(cnt)
n = num[0]
for i in range(1, len(num)):
    n -= num[i]
print(n)