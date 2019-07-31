

def dajimu(N, L, W):
    sl = [(l, w) for l, w in zip(L, W)]
    sl = sorted(sl, key=lambda x: x[0])
    res = [1]*N
    wet = []
    for i, (l, w) in enumerate(sl):
        wet.append(w)
        for j in range(i):
            if res[i] < res[j] + 1 and w*7 >= wet[j]:
                res[i] = res[j] + 1
                wet[i] = wet[j] + w
    return max(res)

N = 10
L = [1,2,10,4,5,6,7,8,9,3]
W = [1,1,10,1,1,1,1,1,1,1]

print(dajimu(N, L, W))