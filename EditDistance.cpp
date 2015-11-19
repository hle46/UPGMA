#include <iostream>

using namespace std;
int minDistance1D(string word1, string word2) {
	int l1 = word1.length(), l2 = word2.length();
        int dp[l2+1];
        for(int i=0;i<=l2;i++){
            dp[i] = i;
        }
        
        for(int j=1;j<=l1;j++){
            int upperleft = dp[0];
            dp[0] = j;
            for( int k=1;k<=l2;k++){
                int upper = dp[k];
                if(word1[j-1]==word2[k-1]){
                    dp[k] = upperleft;
                }else{
                    dp[k] = min(upperleft, min(dp[k],dp[k-1])) + 1;
                }
                upperleft = upper;
            }
        }
        return dp[l2];
}
int minDistance2D(string word1, string word2) {
        int n1 = word1.size(), n2 = word2.size();
        int dp[n1 + 1][n2 + 1];
        for (int i = 0; i <= n1; ++i) dp[i][0] = i;
        for (int i = 0; i <= n2; ++i) dp[0][i] = i;
        for (int i = 1; i <= n1; ++i) {
            for (int j = 1; j <= n2; ++j) {
                if (word1[i - 1] == word2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = min(dp[i - 1][j - 1], min(dp[i - 1][j], dp[i][j - 1])) + 1;
                }
            }
        }
        return dp[n1][n2];
}

int main() {
    cout << "Hello World!" << endl;
	cout << minDistance2D("bad huy","cat guy") << endl;
	cout << minDistance1D("bad huy","cat guy") << endl;
    cin.get();
    return 0;
}
