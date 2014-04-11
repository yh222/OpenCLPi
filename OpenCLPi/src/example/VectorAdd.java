package example;

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
 * Simple JavaCL example that adds two vectors.
 *
 * @author Mark Utting
 */
public class VectorAdd {

	public static void main(String[] args) throws Exception {
		// we can list all available platforms and devices.
		for (CLPlatform p : JavaCL.listPlatforms()) {
			System.out.println("CLPlatform: " + p.getName() + " from " + p.getVendor());
			for (CLDevice dev : p.listAllDevices(false)) {
				System.out.println("  CLDevice: " + dev.getName()
						+ " has " + dev.getMaxComputeUnits() + " compute units");
			}
		}

		// choose the platform and device with the most compute units
		CLContext context = JavaCL.createBestContext(new CLPlatform.DeviceFeature[] { CLPlatform.DeviceFeature.GPU /*.MaxComputeUnits */ });
		System.out.println("best context has device[0]=" + context.getDevices()[0]);

		String src = "\n" +
        "__kernel void vadd(              \n" +
        "   __global const int* a,        \n" +
        "   __global const int* b,        \n" +
        "   __global int* output)         \n" +
        "{                                \n" +
        "   int i = get_global_id(0);     \n" +
        "   output[i] = a[i] + b[i];      \n" +
        "}                                \n";

		CLProgram program = context.createProgram(src).build();
		CLKernel kernel = program.createKernel("vadd");

		// Allocate OpenCL-hosted memory for inputs and output
		final int length = 10000;
		CLBuffer<Integer> memIn1 = context.createIntBuffer(CLMem.Usage.Input, length);
		CLBuffer<Integer> memIn2 = context.createIntBuffer(CLMem.Usage.Input, length);
		CLBuffer<Integer> memOut = context.createIntBuffer(CLMem.Usage.Output, length);

		// Bind these memory objects to the arguments of the kernel
		kernel.setArgs(memIn1, memIn2, memOut);

		CLQueue queue = context.createDefaultQueue();
		/// Grab (map) the input buffers to fill them with data
		Pointer<Integer> a = memIn1.map(queue, MapFlags.Write);
		Pointer<Integer> b = memIn2.map(queue, MapFlags.Write);
		// Fill the mapped input buffers with data
		for (int i = 0; i < length; i++) {
		    a.setInt(i);
		    b.setInt(length - i);
		}
		/// Release the input buffers
		memIn1.unmap(queue, a);
		memIn2.unmap(queue, b);

		// Execute kernels with global size = length
		//                  and workgroup size = 1
		final long time0 = System.nanoTime();
		kernel.enqueueNDRange(queue, new int[]{length}, new int[]{1});

		// Wait for all commands to be completed
		queue.finish();
		final long time1 = System.nanoTime();
		System.out.println("Done in " + (time1 - time0) / 1000 + " microseconds");
		
		// Copy the OpenCL output array back to host RAM
		Pointer<Integer> output = memOut.read(queue);

		// Print out the results, or check that they are correct.
		for (int i = 0; i < length; i++) {
			System.out.println("output[" + i + "] = " + output.get(i));
			assert output.get(i) == length;
		}
	}
}
