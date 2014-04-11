package pi;

import org.bridj.Pointer;

import com.nativelibs4java.opencl.CLBuffer;
import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLDevice;
import com.nativelibs4java.opencl.CLKernel;
import com.nativelibs4java.opencl.CLMem;
import com.nativelibs4java.opencl.CLPlatform;
import com.nativelibs4java.opencl.CLProgram;
import com.nativelibs4java.opencl.CLQueue;
import com.nativelibs4java.opencl.JavaCL;
import com.nativelibs4java.opencl.CLMem.MapFlags;

/**
 * Simple JavaCL example that estimates PI by firing darts at a dart board.
 * This is based on the HADOOP PI estimation example.
 * 
 * This uses JavaCL 1.0.0-RC1, which uses the new BridJ API,
 * with the generic buffers, like CLBuffer&lt;Integer&gt;.
 *
 * @author Mark Utting
 */
public class Pi {

	/**
	 * A dummy Java-version of our kernel.
	 * This is useful so that we can test and debug it in Java first.
	 * 
	 * @param seeds one integer seed for each task (work item).
	 * @param repeats the number of darts each task must throw.
	 * @param output one integer output cell for each task
	 * @param gid dummy global id, only needed in the Java API, not the OpenCL version.
	 *            (delete this parameter when you translate this to an OpenCL kernel).
	 */
	public static void dummyThrowDarts(int[] seeds, int repeats, int[] output, int gid) {
		// int gid = get_global_id(0);  // this is how we get the gid in OpenCL.
		int rand = seeds[gid];
		for (int iter = 0; iter < repeats; iter++) {
			// TODO: write this code
		}
	}

	public static void main(String[] args) throws Exception {
		if (args.length != 3) {
			System.err.println("Usage: pi tasks workgroupsize repeats");
			System.err.println("      (tasks must be a multiple of workgroupsize)");
			System.exit(1);
		}
		final int tasks = Integer.decode(args[0]);
		final int wgSize = Integer.decode(args[1]);
		final int repeats = Integer.decode(args[2]);
		// we can list all available platforms and devices.
		for (CLPlatform p : JavaCL.listPlatforms()) {
			System.out.println("CLPlatform: " + p.getName() + " from " + p.getVendor());
			for (CLDevice dev : p.listAllDevices(false)) {
				System.out.println("  CLDevice: " + dev.getName()
						+ " has " + dev.getMaxComputeUnits() + " compute units");
			}
		}

		// choose the platform and device with the most compute units
		CLContext context = JavaCL.createBestContext();
		System.out.println("best context has device[0]=" + context.getDevices()[0]);

		CLQueue queue = context.createDefaultQueue();

		// Allocate OpenCL-hosted memory for inputs and output
		CLBuffer<Integer> memIn1 = context.createIntBuffer(CLMem.Usage.Input, tasks);
		CLBuffer<Integer> memOut = context.createIntBuffer(CLMem.Usage.Output, tasks);

		// Map input buffers to populate them with some data
		Pointer<Integer> a = memIn1.map(queue, MapFlags.Write);
		// Fill the mapped input buffers with data
		for (int i = 0; i < tasks; i++) {
		    a.setInt(i);
		}
		// Unmap input buffers
		memIn1.unmap(queue, a);

		String src = "\n"
			+ "__kernel void throwDarts(\n"
			+ " TODO: copy your dummyThrowDarts code here and turn it into OpenCL code."
		    + "}\n";

		CLProgram program = context.createProgram(src).build();
		CLKernel kernel = program.createKernel("throwDarts", memIn1, repeats, memOut);

		// Execute the kernel with global size = dataSize and workgroup size = 1
		System.out.println("Starting with " + tasks + " tasks, each doing " + repeats + " repeats.");
		System.out.flush();
		final long time0 = System.nanoTime();
		kernel.enqueueNDRange(queue, new int[]{tasks}, new int[]{wgSize});

		// Wait for all operations to be performed
		queue.finish();
		final long time1 = System.nanoTime();
		System.out.println("Done in " + (time1 - time0) / 1000 + " microseconds");
		
		// Copy the OpenCL-hosted output array back to RAM
		// We could do this via map;take-local-copy;unmap, but read does all that for us.
		Pointer<Integer> output = memOut.read(queue);
		
		// Analyze the results and calculate PI
		long inside = 0;
		long total = (long) tasks * repeats;
		for (int i = 0; i < tasks; i++) {
			//System.out.println("task i: " + i + " gives " + output.get(i));
			inside += output.get(i);
		}
		final double pi = 4.0 * inside / total;
		System.out.println("Estimate PI = " + inside + "/" + total + " = " + pi);
	}
}
