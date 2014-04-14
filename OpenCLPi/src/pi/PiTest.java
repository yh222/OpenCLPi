package pi;

import org.junit.Test;

public class PiTest {
	@Test
	public void testone(){
		int[] results = new int[6];
		Pi.dummyThrowDarts(new int[] {0,1,2,3,4,5}, 1000, results, 0);
		assert(results[0]==750);
		assert(results[1]==0);
		assert(results[2]==0);
		assert(results[3]==0);
		assert(results[4]==0);
		assert(results[5]==0);
	}
	
	@Test
	public void testtwo(){
		int[] results = new int[6];
		Pi.dummyThrowDarts(new int[] {0,1,2,3,4,5}, 1000, results, 5);
		assert(results[0]==0);
		assert(results[1]==0);
		assert(results[2]==0);
		assert(results[3]==0);
		assert(results[4]==0);
		assert(results[5]==750);
	}	
	
}
