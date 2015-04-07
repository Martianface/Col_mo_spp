import profile


def profileTest():
    Total = 1
    for i in range(100):
        Total = Total * (i + 1)
        print Total
    return Total
if __name__ == "__main__":
    # profile.run("profileTest()")
    profile.run("profileTest()", "testprof")
