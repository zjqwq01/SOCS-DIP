import ast
import sys

def provinces(isConnected):
    n = len(isConnected)
    visited = [False] * n
    sys.setrecursionlimit(1_000_000)

    def dfs(i: int):
        visited[i] = True
        for j in range(n):
            if isConnected[i][j] == 1 and not visited[j]:
                dfs(j)

    cnt = 0
    for i in range(n):
        if not visited[i]:
            cnt += 1
            dfs(i)
    return cnt

if __name__ == "__main__":
    # 逐行读取，直到方括号数量配平（支持粘贴多行矩阵）
    s = ""
    while True:
        try:
            line = input()
        except EOFError:
            break
        s += line.strip()
        if s and s.count('[') == s.count(']'):
            break

    mat = ast.literal_eval(s)
    print(provinces(mat))
