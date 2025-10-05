import random
import math
from typing import List

EPSIL = 0.0000000001

def get_mat(n: int) -> List[List[float]]:
    mat = []
    for _ in range(n):
        row = list(map(float, input().split()))
        if len(row) != n: raise Exception(f"{n}*{n}이 아님")
        mat.append(row)

    if len(mat) != n: raise Exception(f"{n}*{n}이 아님")
    return mat



def show_mat(mat: List[List[float]], p: int = 6):
    f = "{:." + str(p) + "f}"
    for r in mat:
        print("  ".join(f.format(x) for x in r))



def det_gauss(m: List[List[float]]) -> float:
    a = [r[:] for r in m]
    n = len(a)
    if n == 0:
        return 1.0
    d = 1.0
    sw = 0
    for i in range(n):
        piv = max(range(i, n), key=lambda r: abs(a[r][i]))
        if abs(a[piv][i]) < EPSIL:
            return 0.0
        if piv != i:
            a[i], a[piv] = a[piv], a[i]
            sw += 1
        p = a[i][i]
        d *= p
        for r in range(i + 1, n):
            if a[r][i] == 0:
                continue
            f = a[r][i] / p
            for c in range(i, n):
                a[r][c] -= f * a[i][c]
    return -d if sw % 2 else d



def minor(m: List[List[float]], sr: int, sc: int) -> List[List[float]]:
    return [[m[i][j] for j in range(len(m)) if j != sc] for i in range(len(m)) if i != sr]



def inv_det(m: List[List[float]]) -> List[List[float]]:
    n = len(m)

    if any(len(r) != n for r in m):
        raise Exception("정방행렬 X")
    
    d = det_gauss(m)
    if abs(d) < EPSIL:
        raise Exception("역행렬없음")
    
    cof = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            cof[i][j] = ((-1) ** (i + j)) * det_gauss(minor(m, i, j))

    adj = [[cof[j][i] for j in range(n)] for i in range(n)]
    
    return [[adj[i][j] / d for j in range(n)] for i in range(n)]



def inv_gauss_jod(m: List[List[float]]) -> List[List[float]]:
    n = len(m)
    if any(len(r) != n for r in m):
        raise Exception("정방행렬X")
    A = [r[:] for r in m]
    aug = [A[i] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(aug[r][c]))
        if abs(aug[piv][c]) < EPSIL:
            raise Exception("역행렬 없음")
        if piv != c:
            aug[c], aug[piv] = aug[piv], aug[c]
        p = aug[c][c]
        aug[c] = [x / p for x in aug[c]]
        for r in range(n):
            if r == c:
                continue
            f = aug[r][c]
            if abs(f) > 0:
                aug[r] = [aug[r][k] - f * aug[c][k] for k in range(2 * n)]
    return [r[n:] for r in aug]



def equl(A: List[List[float]], B: List[List[float]]) -> bool:
    if len(A) != len(B):
        return False
    for i in range(len(A)):
        for j in range(len(A[i])):
            if abs(A[i][j] - B[i][j]) > EPSIL:
                return False
    return True


def main():
    print("--- 행렬 역행렬 프로그램 ---")

    n = int(input("행렬 크기 입력: "))

    print(f"{n}x{n} 행렬의 행:")

    while True:
        try:
            A = get_mat(n)
            break
        except Exception as e:
            print("오류:", e)
            print("재입력")
            continue

    print("\n입력 행렬:")
    show_mat(A)

    print("\n--- 행렬식 방식 ---")
    try:
        inv1 = inv_det(A)
        show_mat(inv1)
    except Exception as e:
        print("오류:", e)
        inv1 = None

    print("\n--- 가우스조던 방식 ---")
    try:
        inv2 = inv_gauss_jod(A)
        show_mat(inv2)
    except Exception as e:
        print("오류:", e)
        inv2 = None

    print("\n--- 비교 ---")
    if inv1 and inv2:
        print("두 결과가 동일한가?", "예" if equl(inv1, inv2) else "아니오")
    else:
        print("불가능한 행렬 존재")


if __name__ == "__main__":
    main()
