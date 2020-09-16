public static void main(String[] args){
    for(int i = -5; i < 33; i++){
        System.out.println(i + ": " + toBinary(i));
        System.out.println(i);
        //always another way
        System.out.println(i + ": " + Integer.toBinaryString(i));
    }
    assert val != null : "Failed precondition all0sAnd1s. parameter cannot be null";
    boolean all = true;
    int i = 0;
    char c;
    
    while(all && i < val.length()){
        c = val.charAt(i);
        all = c == '0' || c == '1';
        i++;
    }
    return all;
}
