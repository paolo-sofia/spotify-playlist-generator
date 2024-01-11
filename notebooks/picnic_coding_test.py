def solution(n):
    # handle base cases for the Oth, 1st and 2nd elements
    if n == 0:
        return 0
    if n in [1, 2]:
        return 1

    previous_num: int = 1
    second_previous_num: int = 1
    for _ in range(3, n + 1):
        previous_sum: int = sum(int(digit) for digit in str(previous_num))
        second_previous_sum: int = sum(int(digit) for digit in str(second_previous_num))
        value = previous_sum + second_previous_sum
        previous_num = second_previous_num
        second_previous_sum = value

    return value


if __name__ == "__main__":
    assert solution(0) == 0
    assert solution(1) == 1
    assert solution(2) == 1
    assert solution(3) == 2
    assert solution(4) == 3
    assert solution(5) == 5
    assert solution(6) == 8
    assert solution(7) == 13
    assert solution(8) == 12
    assert solution(9) == 7
