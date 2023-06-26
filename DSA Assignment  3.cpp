//******* Assignment Question 1 :

# include<iostream>
# include<vector>
using namespace std;

class Solution {
public:
    int threeSumClosest(vector<int>& nums, int target) {
        sort(nums.begin(), nums.end());
        int n = nums.size();
        int closest_sum = nums[0] + nums[1] + nums[2];
        for(int i=0; i<n-2; i++){
            int left = i+1, right = n-1;
            while(left<right){
                int sum = nums[i] + nums[left] + nums[right];
                if(sum == target){
                    return sum;
                }
                else if(sum<target){
                    left++;
                }
                else{
                    right--;
                }
                if(abs(sum-target) < abs(closest_sum-target)){
                    closest_sum = sum;
                }
            }
        }
        return closest_sum;
    }
};

int main(){
    Solution s ;
    vector<int> v;
     v={-1,2,1,-4};
     vector<int> result= s.threeSumClosest (v, 1);
     for(int i: result)
     {
          cout<<i<<endl;
     }
     return 0;
}

//******* Assignment Question 2 :

# include<iostream>
# include<vector>
using namespace std;

class Solution {
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
         vector<vector<int> > res;
        
        if (nums.empty())
            return res;
        int n = nums.size(); 
        sort(nums.begin(),nums.end());
    
        for (int i = 0; i < n; i++) {
        
            for(int j=i+1; j<n; j++){
                int target_2 = target - nums[j] - nums[i];
            
                int front = j + 1;
                int back = n - 1;
            
                while(front < back) {
                
                    int two_sum = nums[front] + nums[back];
                
                    if (two_sum < target_2) front++;
                
                    else if (two_sum > target_2) back--;
                
                    else {
                    
                        vector<int> quadruplet(4, 0);
                        quadruplet[0] = nums[i];
                        quadruplet[1] = nums[j];
                        quadruplet[2] = nums[front];
                        quadruplet[3] = nums[back];
                        res.push_back(quadruplet);
                    
                        // Processing the duplicates of number 3
                        while (front < back && nums[front] == quadruplet[2]) ++front;
                    
                        // Processing the duplicates of number 4
                        while (front < back && nums[back] == quadruplet[3]) --back;
                
                    }
                }
                
                // Processing the duplicates of number 2
                while(j + 1 < n && nums[j + 1] == nums[j]) ++j;
            }
        
            // Processing the duplicates of number 1
            while (i + 1 < n && nums[i + 1] == nums[i]) ++i;
        
        }
    
        return res;
    }
};

int main(){
    Solution s ;
    vector<int> v;
     v={1,0,-1,0,-2,2};
     vector<int> result= s.fourSum (v, 0);
     for(int i: result)
     {
          cout<<i<<endl;
     }
     return 0;
}

//******* Assignment Question 3 :

# include<iostream>
# include<vector>
using namespace std;

class Solution {
public:
    void nextPermutation(vector<int>& nums) {
    int n=nums.size(), k, l;
    for(k=n-2; k>=0; k--){
        if(nums[k]<nums[k+1]){
            break;
        }
    }
    if(k<0){
        reverse(nums.begin(), nums.end());
    }
    else{
        for(l=n-1; l>k; l--){
            if(nums[l]>nums[k]){
                break;
            }
        }
        swap(nums[k], nums[l]);
        reverse(nums.begin()+k+1, nums.end());
    }
    }
};

int main(){
    Solution s ;
    vector<int> v;
     v={1,2,3};
     vector<int> result= s.nextPermutation (v);
     for(int i: result)
     {
          cout<<i<<endl;
     }
     return 0;
}

//******* Assignment Question 4 :

# include<iostream>
# include<vector>
using namespace std;

class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
        int s = 0;
        int e = nums.size()-1;
        while(s<=e){
            int mid = s+(e-s)/2;
            if(nums[mid]==target){
                return mid;
            }
            if(target<nums[mid]){
                e = mid-1;
            }
            else{
                s = mid+1;
            }
        }
        return s;
    }
};

int main(){
    Solution s ;
    vector<int> v;
     v={1,3,5,6};
     vector<int> result= s.searchInsert (v, 5);
     for(int i: result)
     {
          cout<<i<<endl;
     }
     return 0;
}

//******* Assignment Question 5 :

# include<iostream>
# include<vector>
using namespace std;

class Solution {
public:
    vector<int> plusOne(vector<int>& digit){
        int n = digit.size();
        for(int i=n-1; i>=0; i--){
            if(digit[i]<9){
                digit[i]++;
                return digit;
            }
            else{
                digit[i] = 0;
            }
        }
        digit.push_back(0);     
        digit[0] = 1;
        return digit;
    }
};

int main(){
    Solution s ;
    vector<int> v;
     v={1,2,3};
     vector<int> result= s.plusOne (v);
     for(int i: result)
     {
          cout<<i<<endl;
     }
     return 0;
}

//******* Assignment Question 6 :

# include<iostream>
# include<vector>
using namespace std;

class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int ans = 0;
        for(int num:nums){
            ans = ans^num;
        }
        return ans;
    }
};

int main(){
    Solution s ;
    vector<int> v;
     v={2,2,1};
     vector<int> result= s.singleNumber (v);
     for(int i: result)
     {
          cout<<i<<endl;
     }
     return 0;
}

//******* Assignment Question 7 :

# include<iostream>
# include<vector>
using namespace std;


class Solution {
 public:
  vector<string> findMissingRanges(vector<int>& nums, int lower, int upper) {
    if (nums.empty())
      return {getRange(lower, upper)};

    vector<string> ans;

    if (nums.front() > lower)
      ans.push_back(getRange(lower, nums.front() - 1));

    for (int i = 1; i < nums.size(); ++i)
      if (nums[i] > nums[i - 1] + 1)
        ans.push_back(getRange(nums[i - 1] + 1, nums[i] - 1));

    if (nums.back() < upper)
      ans.push_back(getRange(nums.back() + 1, upper));

    return ans;
  }

 private:
  string getRange(int lo, int hi) {
    if (lo == hi)
      return to_string(lo);
    return to_string(lo) + "->" + to_string(hi);
  }
};

int main(){
    Solution s ;
    vector<int> v;
     v={0,1,3,50,75};
     vector<int> result= s.findMissingRanges (v, 0, 99);
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
    bool canAttendMeetings(vector<Interval>& intervals) {
        sort(intervals.begin(), intervals.end(), [](const Interval& x, const Interval& y) { return x.start < y.start; });
        for (int i = 1; i < intervals.size(); ++i) {
            if (intervals[i].start < intervals[i - 1].end) {
                return false;
            }
        }
        return true;
    }
};

int main(){
    Solution s ;
    vector<int> v;
     v={[0,30],[5,10],[15,20]};
     vector<int> result= s.canAttendMeetings (v);
     for(int i: result)
     {
          cout<<i<<endl;
     }
     return 0;
}