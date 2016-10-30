# Leetcode

## 1.Serialize and deserialize binary tree

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class Codec {
    private static final String NN = "x";
    private static final String splitor = ",";

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        StringBuilder sb = new StringBuilder();
        buildString(root, sb);
        return sb.toString();
    }
    
    private void buildString(TreeNode root, StringBuilder sb) {
        if (root == null) {
            sb.append(NN).append(splitor);
        } else {
            sb.append(root.val).append(splitor);
            buildString(root.left, sb);
            buildString(root.right, sb);
        }
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        Deque<String> nodes = new LinkedList<>();
        nodes.addAll(Arrays.asList(data.split(splitor)));
        return buildTree(nodes);
    }
    
    private TreeNode buildTree(Deque<String> nodes) {
        String val = nodes.remove();
        if (val.equals(NN)) {
            return null;
        } else {
            TreeNode node = new TreeNode(Integer.valueOf(val));
            node.left = buildTree(nodes);
            node.right = buildTree(nodes);
            return node;
        }
    }
}
```

## 2.Intersection of two arrays

```java
public class Solution {
    public int[] intersection(int[] nums1, int[] nums2) {
        HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
        List<Integer> result = new ArrayList<Integer>();
        int len1 = nums1.length;
        int len2 = nums2.length;
        for (int i = 0; i < len1; i ++) {
            if (map.containsKey(nums1[i])) {
                continue;
            } else {
                map.put(nums1[i], 1);
            }
        }
        for (int j = 0; j < len2; j ++) {
            if (map.containsKey(nums2[j]) && map.get(nums2[j]) <= 1) {
                result.add(nums2[j]);
                map.put(nums2[j], map.get(nums2[j]) + 1);
            }
        }
        int length = result.size();
        int[] res = new int[length];
        for (int k = 0; k < length; k++) {
            res[k] = result.get(k);
        }
        return res;      
    }
}
```

## 3.Intersection of two arrays II

```java
public class Solution {
    public int[] intersect(int[] nums1, int[] nums2) {
        HashMap<Integer, Integer> map = new HashMap<>();
        List<Integer> result = new ArrayList<Integer>();
        int len1 = nums1.length;
        int len2 = nums2.length;
        for (int i = 0; i < len1; i ++) {
            if (map.containsKey(nums1[i])) {
                map.put(nums1[i], map.get(nums1[i]) + 1);
            } else {
                map.put(nums1[i], 1);
            }
        }
        for (int j = 0; j < len2; j ++) {
            if (map.containsKey(nums2[j]) && map.get(nums2[j]) > 0) {
                result.add(nums2[j]);
                map.put(nums2[j], map.get(nums2[j]) - 1);
            } else {
                continue;
            }
        }
        int length = result.size();
        int[] res = new int[length];
        for (int i = 0; i < length; i ++) {
            res[i] = result.get(i);
        }
        return res;
    }
}
```

## 4.Third maximum number

```java
public class Solution {
    public int thirdMax(int[] nums) {
    long max = Long.MIN_VALUE;
    mid = max, min = max;
    for (int ele : nums) {
      if (ele > max) {
        min = mid;
        mid = max;
        max = ele;
      } else if (max > ele && ele > mid) {
        min = mid;
        mid = ele;
      } else if (mid > ele && ele > min) {
        min = ele;
      }
    }
    return (int)(min != Long.MIN_VALUE) ? min : max;
    }
}
```

## 5.Reverse vowels of a string

```java
public class Solution {
    public String reverseVowels(String s) {
        if (s == null || s.length() == 0) {
            return s;
        }
        String vowels = "aeiouAEIOU";
        char[] chars = s.toCharArray();
        int start = 0;
        int end = s.length() - 1;
        while (start < end) {
            while (start < end && !vowels.contains(chars[start]+"")) {
                start++;
            }
            while ( start < end && !vowels.contains(chars[end]+"")) {
                end--;
            }
            char temp = chars[start];
            chars[start] = chars[end];
            chars[end] = temp;
            start++;
            end--;
        }
        return new String(chars);
    }
}
```

## 6.Missing number

```java
public class Solution {
    public int missingNumber(int[] nums) {
        int res = nums.length;
        for (int i = 0;i < nums.length;i++) {
            res ^= i;
            res ^= nums[i];
        }
        return res;
    }
}
```

