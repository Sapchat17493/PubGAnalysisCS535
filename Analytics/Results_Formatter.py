import argparse
import os
import sys
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(description="Plot Data for CS535 Project PUBG Analyses")
    opt_args = parser._action_groups.pop()
    req_args = parser.add_argument_group('required arguments')
    opt_args = parser.add_argument_group('optional arguments')
    req_args.add_argument("-d", "--dirc", help="Data Directory", action="store", required=True)

    if len(sys.argv) < 1:
        parser.print_help()
        sys.exit(1)

    return (parser.parse_args())


def read_file(filepath):
    with open(filepath) as fp:
        line = fp.readline()
        cnt = 1
        while line:
            print("Line {}: {}".format(cnt, line.strip()))
            line = fp.readline()
            cnt += 1


def read_and_plot_case1(filepath):
    d = {}
    with open(filepath) as fp:
        line = fp.readline()
        while line:
            line = fp.readline()
            l = line.split(",")
            d[l[0]] = l[1:]


def read_and_plot_case2(filepath):
    X = []
    Y = []
    with open(filepath) as fp:
        line = fp.readline()
        while line:
            line = fp.readline()
            line = line[1:len(line) - 2]
            l = line.split(", ")
            if len(l) > 1:
                X.append(l[0])
                Y.append(l[1])


def read_and_plot_case0345(filepath, delim):
    X = []
    Y = []
    with open(filepath) as fp:
        line = fp.readline()
        while line:
            line = fp.readline()
            l = line.split(delim)
            if len(l) > 1:
                X.append(l[0])
                Y.append(l[1].strip())


def read_and_plot_case6(filepath):
    X = []
    Y = []
    with open(filepath) as fp:
        line = fp.readline()
        while line:
            line = fp.readline()
            line = line[3:len(line) - 2]
            l = line.split(", ")
            if len(l) > 1:
                X.append(l[0])
                Y.append(l[1])


def read_and_plot_case7(filepath):
    rows, cols = (101, 101)
    arr = [[0] * cols] * rows
    with open(filepath) as fp:
        line = fp.readline()
        while line:
            line = fp.readline()
            l = line.split(",")
            if len(l) > 1:
                ind = l[0].split(":")
                if len(ind) > 1:
                    arr[int(ind[0])][int(ind[1])] = l[1].strip()


def table_XY(X, Y, xlabel, ylabel, label):
    plt.scatter(X, Y, color='red', label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    args = get_args()

    dirc = args.dirc
    cnt = 0
    avoid_list = ['']

    for (dirpath, dirnames, filenames) in os.walk(dirc):
        for f in filenames:
            if dirpath == 'Analysis_Results_PUBG':
                if f == 'WeaponKillDist.txt':  # Case 0
                    read_and_plot_case0345(dirpath + "/" + f, delim="\t")  # plot kill distance here
            else:
                if f != '_SUCCESS':
                    dir = dirpath.split('/')[1]
                    filepath = dirpath + "/" + f
                    if dir == 'survival_team_size_placement':  # Case 1
                        read_and_plot_case1(filepath)
                    elif dir == 'survivors(For a Given time slot)':  # Case 2
                        read_and_plot_case2(filepath)
                    elif dir == 'average_damage (Damage by team placement, i.e position)':  # Case 3
                        read_and_plot_case0345(filepath, delim=",")
                    elif dir == 'median_place_only':  # Case 4
                        read_and_plot_case0345(filepath, delim=",")
                    elif dir == 'survival_times(Survival Times by team placement)':  # Case 5
                        read_and_plot_case0345(filepath, delim=",")
                    elif dir == 'weapon_kills(Number of kills per weapon)':  # Case 6
                        read_and_plot_case6(filepath)
                    elif dir == 'median_place_size':  # Case 7
                        read_and_plot_case7(filepath)
