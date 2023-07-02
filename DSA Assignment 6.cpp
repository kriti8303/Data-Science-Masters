//******* Assignment Question 1 :

# include<iostream>
# include<vector>
using namespace std;

class Solution {
public:
    vector<int> diStringMatch(string s) {
        int n = s.size();
        int i = 0;
        int j = n;

        vector<int> ans;
        for(int k=0; k<=n; k++){
            if(s[k] == 'I'){
                ans.push_back(i);
                i++;
            }
            else{
                ans.push_back(j);
                j--;
            }
        }
        if(i == j){
            ans.push_back(i);
        }
        return ans;
    }
};

int main(){
    Solution s ;
    vector<string> v;
     v={"IDID"};
     vector<int> result= s.diStringMatch (v);
     for(int i: result)
     {
          cout<<i<<endl;
     }
     return 0;
}


//******* Assignment Question 2:

# include<iostream>
# include<vector>
using namespace std;

class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int row = matrix.size();
        int col = matrix[0].size();
        int start = 0;
        int end = row*col-1;
        int mid = start +(end-start)/2;
        while(start<=end){
            int element = matrix[mid/col][mid%col];
            if(element == target){
                return 1;
            }
            else if(element < target){
                start = mid+1;
            }
            else{
                end = mid-1;
            }
            mid = start + (end-start)/2;
        }
        return 0;
    }
};

int main(){
    Solution s ;
    vector<int> v;
     v={{1,3,5,7},{10,11,16,20},{23,30,34,60}};
     vector<int> result= s.searchMatrix (v, 3);
     for(int i: result)
     {
          cout<<i<<endl;
     }
     return 0;
}


//******* Assignment Question 3:

# include<iostream>
# include<vector>
using namespace std;

class Solution {
public:
    bool validMountainArray(vector<int>& arr) {
        if(arr.size()<3){
            return false;
        }
        int s = 0;
        int e = arr.size()-1;
        while(s+1 < arr.size()-1 && arr[s] < arr[s+1]){
            s++;
        }
        while(e-1 > 0 && arr[e] < arr[e-1]){
            e--;
        }
        return s==e;
    }
};

int main(){
    Solution s ;
    vector<int> v;
     v={2,1};
     vector<int> result= s.validMountainArray (v);
     for(int i: result)
     {
          cout<<i<<endl;
     }
     return 0;
}



//******* Assignment Question 4:

# include<iostream>
# include<vector>
using namespace std;

class Solution {
public:
    int findMaxLength(vector<int>& nums) {
        int sum = 0;
        int maxLen = 0;
        unordered_map<int, int> seen{{0, -1}};
        for(int i=0; i<nums.size(); i++){
            sum += nums[i]==1 ? 1 : -1;
            if(seen.count(sum)){
                maxLen = max(maxLen, i-seen[sum]);
            }
            else{
                seen[sum] = i;
            }
        }
        return maxLen;
    }
};

int main(){
    Solution s ;
    vector<int> v;
     v={0,1};
     vector<int> result= s.findMaxLength (v);
     for(int i: result)
     {
          cout<<i<<endl;
     }
     return 0;
}



//******* Assignment Question 5:

# include<iostream>
# include<vector>
using namespace std;

class Solution {
public:
    int minProductSum(vector<int>& A, vector<int>& B) {
        sort(begin(A), end(A));
        sort(begin(B), end(B), greater<>());
        int ans = 0;
        for (int i = 0; i < A.size(); ++i) ans += A[i] * B[i];
        return ans;
    }
};

int main(){
    Solution s ;
    vector<int> nums1, nums2;
     nums1 = {2,1,4,5,7}, nums2 = {3,2,4,8,6};
     vector<int> result= s.minProductSum (nums1, nums2);
     for(int i: result)
     {
          cout<<i<<endl;
     }
     return 0;
}


//******* Assignment Question 6:

# include<iostream>
# include<vector>
using namespace std;

class Solution {
public:
    vector<int> findOriginalArray(vector<int>& changed) {
        map<int, int> mp;
        vector<int> ans;
        int n = changed.size();
        if(n%2){
            return ans;
        }
        for(auto x : changed){
            mp[x]++;
        }
        sort(changed.begin(), changed.end());
        for(auto x : changed){
            if(mp[x]==0){
                continue;
            }
            if(mp[2*x]==0){
                return {};
            }
            if(mp[x] && mp[2*x]){
                mp[2*x]--;
                ans.push_back(x);
                mp[x]--;
            }
        }
        return ans;
    }
};

int main(){
    Solution s ;
    vector<int> v;
     v={1,3,4,2,6,8};
     vector<int> result= s.findOriginalArray (v);
     for(int i: result)
     {
          cout<<i<<endl;
     }
     return 0;
}


//******* Assignment Question 7:

# include<iostream>
# include<vector>
using namespace std;

class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        vector<int> ans;
        int row=matrix.size();
        int col=matrix[0].size();
        int count=0;
        int total=row*col;
        int startingRow=0;
        int startingCol=0;
        int endingRow=row-1;
        int endingCol=col-1;
        while(count<total){
            for(int index=startingCol; count<total && index<=endingCol; index++){
                ans.push_back(matrix[startingRow][index]);
                count++;
            }
            startingRow++;

            for(int index=startingRow; count<total && index<=endingRow; index++){
                ans.push_back(matrix[index][endingCol]);
                count++;
            }
            endingCol--;

            for(int index=endingCol; count<total && index>=startingCol; index--){
                ans.push_back(matrix[endingRow][index]);
                count++;
            }
            endingRow--;

            for(int index=endingRow; count<total && index>=startingRow; index--){
                ans.push_back(matrix[index][startingCol]);
                count++;
            }
            startingCol++;
        }
        return ans;
    }
};

int main(){
    Solution s ;
    vector<string> v;
     v=3;
     vector<int> result= s.spiralOrder (v);
     for(int i: result)
     {
          cout<<i<<endl;
     }
     return 0;
}



//******* Assignment Question 8:

# include<iostream>
# include<vector>
using namespace std;

class Solution {
public:
    vector<vector<int>> multiply(vector<vector<int>>& mat1, vector<vector<int>>& mat2) {
        int r1 = mat1.size(), c1 = mat1[0].size(), c2 = mat2[0].size();
        vector<vector<int>> res(r1, vector<int>(c2));
        unordered_map<int, vector<int>> mp;
        for (int i = 0; i < r1; ++i) {
            for (int j = 0; j < c1; ++j) {
                if (mat1[i][j] != 0) mp[i].push_back(j);
            }
        }
        for (int i = 0; i < r1; ++i) {
            for (int j = 0; j < c2; ++j) {
                for (int k : mp[i]) res[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
        return res;
    }
};

int main(){
    Solution s ;
    vector<int> mat1, mat2;
     mat1 = [[1,0,0],[-1,0,3]], mat2 = [[7,0,0],[0,0,0],[0,0,1]];
     vector<int> result= s.multiply (mat1, mat2);
     for(int i: result)
     {
          cout<<i<<endl;
     }
     return 0;
}
