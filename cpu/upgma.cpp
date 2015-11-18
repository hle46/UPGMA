#include <iostream>
#include <limits>

using namespace std;

struct Tree {
  int num_nodes;
  Tree *left;
  Tree *right;
  float total_length;
  float branch_length[2];
  Tree(int _num_nodes, float _length, Tree *_left, Tree *_right, float length1,
       float length2)
      : num_nodes(_num_nodes), left(_left), right(_right),
        total_length(_length) {
    branch_length[0] = length1;
    branch_length[1] = length2;
  }
};

int getMinIdx(float *mat, int n) {
  float val = numeric_limits<float>::infinity();
  int idx = -1;
  for (int i = 0; i < n; ++i) {
    if (mat[i] != 0.0f && mat[i] < val) {
      idx = i;
      val = mat[i];
    }
  }
  return idx;
}

void update(float *mat, int n, int idx1, int idx2, int num_nodes1,
            int num_nodes2) {
  int total_nodes = num_nodes1 + num_nodes2;
  for (int i = 0; i < n; ++i) {
    float val =
        (mat[n * idx1 + i] * num_nodes1 + mat[n * idx2 + i] * num_nodes2) /
        total_nodes;
    mat[n * idx1 + i] = val;
    mat[n * idx2 + i] = 0.0f;
    mat[n * i + idx1] = val;
    mat[n * i + idx2] = 0.0f;
  }
  mat[n * idx1 + idx1] = 0.0f;
  mat[n * idx1 + idx2] = 0.0f;
  mat[n * idx2 + idx1] = 0.0f;
  mat[n * idx2 + idx2] = 0.0f;
}

void cleanupTree(Tree *tree) {
  // Reach the leaf
  if (tree->left == nullptr && tree->right == nullptr) {
    delete tree;
    return;
  }
  cleanupTree(tree->left);
  cleanupTree(tree->right);
}

void printTree(Tree *tree) {
  // Reach the leaf
  if (tree->left == nullptr && tree->right == nullptr) {
    return;
  }
  cout << "(";
  printTree(tree->left);
  cout << ": " << tree->branch_length[0] << ", ";
  printTree(tree->right);
  cout << ": " << tree->branch_length[1] << ")";
}

int main() {
  const int num_seqs = 7;
  float a[num_seqs][num_seqs]{{0.0f, 19.0f, 27.0f, 8.0f, 33.0f, 18.0f, 13.0f},
                              {19.0f, 0.0f, 31.0f, 18.0f, 36.0f, 1.0f, 13.0f},
                              {27.0f, 31.0f, 0.0f, 26.0f, 41.0f, 32.0f, 29.0f},
                              {8.0f, 18.0f, 26.0f, 0.0f, 31.0f, 17.0f, 14.0f},
                              {33.0f, 36.0f, 41.0f, 31.0f, 0.0f, 35.0f, 28.0f},
                              {18.0f, 1.0f, 32.0f, 17.0f, 35.0f, 0.0f, 12.0f},
                              {13.0f, 13.0f, 29.0f, 14.0f, 28.0f, 12.0f, 0.0f}};

  Tree *nodes[num_seqs];
  for (int i = 0; i < num_seqs; ++i) {
    nodes[i] = new Tree(1, 0.0f, nullptr, nullptr, 0.0f, 0.0f);
  }

  Tree *root;
  for (int remain = num_seqs; remain >= 2; --remain) {
    int idx = getMinIdx((float *)a, num_seqs * num_seqs);
    int idx1 = idx / num_seqs;
    int idx2 = idx % num_seqs;
    if (idx1 > idx2) {
      swap(idx1, idx2);
    }
    float length = a[idx1][idx2];
    root = new Tree(nodes[idx1]->num_nodes + nodes[idx2]->num_nodes, length / 2,
                    nodes[idx1], nodes[idx2],
                    length / 2 - nodes[idx1]->total_length,
                    length / 2 - nodes[idx2]->total_length);
    update((float *)a, num_seqs, idx1, idx2, nodes[idx1]->num_nodes,
           nodes[idx2]->num_nodes);
    nodes[idx1] = root;
  }

  printTree(root);

  // Print the Tree
  cleanupTree(root);

  return 0;
}
