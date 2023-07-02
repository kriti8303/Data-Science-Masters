//******* Assignment Question 1 :

# include<iostream>
# include<vector>
using namespace std;

class Solution {
public:
    vector<vector<int>> construct2DArray(vector<int>& original, int m, int n) {
        if(m*n != original.size()){
            return {};
        }
        vector<vector<int>> ans(m, vector<int>(n, 0));
        for(int i=0; i<original.size(); i++){
            ans[i/n][i%n] = original[i];
        }
        return ans;
    }
};

int main(){
    Solution s ;
    vector<int> v;
     v={1,2,3,4};
     vector<int> result= s.construct2DArray (v, 2, 2);
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
    int arrangeCoins(int n) {
        return(-1 + sqrt(1.0 +4.0 *2.0 *n))/2;
    }
};

int main(){
    Solution s ;
    int v;
     v=5;
     vector<int> result= s.arrangeCoins (v);
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
    vector<int> sortedSquares(vector<int>& nums) {
        vector<int> ans;
        for(int i=0; i<nums.size(); i++){
            ans.push_back(nums[i]*nums[i]);
        }
        sort(ans.begin(), ans.end());
        return ans;
    }
};

int main(){
    Solution s ;
    vector<int> v;
     v={-4,-1,0,3,10};
     vector<int> result= s.sortedSquares (v);
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
    vector<vector<int>> findDifference(vector<int>& nums1, vector<int>& nums2) {
        unordered_set<int> s1, s2;
        vector<vector<int>> answer(2);
        for(auto num : nums1){
            s1.insert(num);
        }
        for(auto num : nums2){
            s2.insert(num);
        }
        for(auto num : s1){
            if(s2.find(num) == s2.end()){
                answer[0].push_back(num);
            }
        }
        for(auto num : s2){
            if(s1.find(num) == s1.end()){
                answer[1].push_back(num);
            }
        }
        return answer;
    }
};

int main(){
    Solution s ;
    vector<int> num1, num2;
     num1 = {1,2,3};
	 num2 = {2,4,6};
     vector<int> result= s.findDifference (num1, num2);
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
    int findTheDistanceValue(vector<int>& arr1, vector<int>& arr2, int d) {
        sort(arr2.begin(), arr2.end());
        int count = 0, low, high;
        for(auto x : arr1){
            low = x-d, high = x+d;
            auto l = lower_bound(arr2.begin(), arr2.end(), low);
            auto h = lower_bound(arr2.begin(), arr2.end(), high);
            if(l==h && (*l != low && *l != high)){
                count++;
            }
        }
        return count;
    }
};

int main(){
    Solution s ;
    vector<int> arr1, arr2;
     arr1 = {4,5,8};
	 arr2 = {10,9,1,8};
     vector<int> result= s.findTheDistanceValue (arr1,arr2, 2);
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
    vector<int> findDuplicates(vector<int>& nums) {
        vector<int> ans;
        sort(nums.begin(), nums.end());
        for(int i=1; i<nums.size(); i++){
            if(nums[i] == nums[i-1]){
                ans.push_back(nums[i]);
            }
        }
        return ans;
    }
};

int main(){
    Solution s ;
    vector<int> v;
     v={4,3,2,7,8,2,3,1};
     vector<int> result= s.findDuplicates (v);
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
    int findMin(vector<int>& nums) {
        if(nums.size()==1){
            return nums[0];
        }
        if(nums[0]<nums.back()){ //if first element is less than last one means array is not r
            return nums[0];
        }
        int ans = INT_MAX;
        int s=0, e = nums.size()-1;
        while(s<=e){
            int mid = s+(e-s)/2;
            if(nums[mid]>=nums[0]){
                s = mid+1;
            }else{
                ans = min(ans, nums[mid]);
                e = mid-1;
            }
        }
        return ans;
    }
};

int main(){
    Solution s ;
    vector<int> v;
     v={3,4,5,1,2};
     vector<int> result= s.findMin (v);
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