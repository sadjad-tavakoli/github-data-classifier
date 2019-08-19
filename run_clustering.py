import sys

from clustering.agglomerative import agglomerative_elbow, agglomerative_clustering

if __name__ == "__main__":
    user_input = sys.argv[1] if len(sys.argv) > 1 else "invalid"

    if user_input == "elbow":
        agglomerative_elbow()

    elif user_input.isdigit():
        agglomerative_clustering(int(user_input))
    else:
        print("\n invalid request")
