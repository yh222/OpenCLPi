package example;

import java.util.Arrays;

import org.bridj.Pointer;

import com.nativelibs4java.opencl.CLBuffer;
import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLDevice;
import com.nativelibs4java.opencl.CLEvent;
import com.nativelibs4java.opencl.CLKernel;
import com.nativelibs4java.opencl.CLMem;
import com.nativelibs4java.opencl.CLPlatform;
import com.nativelibs4java.opencl.CLProgram;
import com.nativelibs4java.opencl.CLQueue;
import com.nativelibs4java.opencl.JavaCL;
import com.nativelibs4java.opencl.CLEvent.CommandExecutionStatus;
import com.nativelibs4java.opencl.CLMem.MapFlags;

/**
 * Experimenting with out-of-order queues.
 *
 * @author Mark Utting
 */
public class OutOfOrder {

	public static void main(String[] args) throws Exception {
		// we can list all available platforms and devices.
		for (CLPlatform p : JavaCL.listPlatforms()) {
			System.out.println("CLPlatform: " + p.getName() + " from " + p.getVendor());
			for (CLDevice dev : p.listAllDevices(false)) {
				System.out.println("  CLDevice: " + dev.getName()
						+ " has " + dev.getMaxComputeUnits() + " compute units");
				System.out.println("maxWorkGroupSize=" + dev.getMaxWorkGroupSize());
				System.out.println("maxWorkItemDimensions=" + dev.getMaxWorkItemDimensions());
				System.out.println("maxWorkItemSizes=" + Arrays.toString(dev.getMaxWorkItemSizes()));
				System.out.println("extensions=" + Arrays.toString(dev.getExtensions()));
			}
		}

		// choose the platform and device with the most compute units
		CLContext context = JavaCL.createBestContext();
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

		// Allocate OpenCL-hosted memory for inputs and output
		final int length = 512 * 10000;
		CLBuffer<Integer> memIn1a = context.createIntBuffer(CLMem.Usage.Input, length);
		CLBuffer<Integer> memIn1b = context.createIntBuffer(CLMem.Usage.Input, length);
		CLBuffer<Integer> memIn2 = context.createIntBuffer(CLMem.Usage.Input, length);
		CLBuffer<Integer> memOuta = context.createIntBuffer(CLMem.Usage.Output, length);
		CLBuffer<Integer> memOutb = context.createIntBuffer(CLMem.Usage.Output, length);

		CLProgram program = context.createProgram(src).build();
		CLKernel kernela = program.createKernel("vadd", memIn1a, memIn2, memOuta);
		CLKernel kernelb = program.createKernel("vadd", memIn1b, memIn2, memOutb);
		
		CLQueue queue = context.createDefaultOutOfOrderQueue();
		/// Grab (map) the input buffers to fill them with data
		Pointer<Integer> aa = memIn1a.map(queue, MapFlags.Write);
		Pointer<Integer> ab = memIn1b.map(queue, MapFlags.Write);
		Pointer<Integer> b = memIn2.map(queue, MapFlags.Write);
		// Fill the mapped input buffers with data
		for (int i = 0; i < length; i++) {
		    aa.set(i, i);
		    ab.set(i, i + 7);
		    b.set(i, length - i);
		}
		/// Release the input buffers
		memIn1a.unmap(queue, aa);
		memIn1b.unmap(queue, ab);
		memIn2.unmap(queue, b);

		// Execute kernels with global size = length
		//                  and workgroup size = 512 (the maximum)
		final long time0 = System.nanoTime();
		CLEvent ea = kernela.enqueueNDRange(queue, new int[]{length}, new int[]{512});
		CommandExecutionStatus statusA = ea.getCommandExecutionStatus();
		CommandExecutionStatus statusB = null;
		System.out.println("status=" + statusA + "," + statusB);
		CLEvent eb = kernelb.enqueueNDRange(queue, new int[]{length}, new int[]{512});
		statusA = ea.getCommandExecutionStatus();
		statusB = eb.getCommandExecutionStatus();
		System.out.println("status=" + statusA + "," + statusB);
		// Wait for all commands to be completed
		queue.finish();
		final long time1 = System.nanoTime();
		statusA = ea.getCommandExecutionStatus();
		statusB = eb.getCommandExecutionStatus();
		System.out.println("status=" + statusA + "," + statusB);
		System.out.println("Done in " + (time1 - time0) / 1000 + " microseconds");
		
		// Copy the OpenCL output array back to host RAM
		Pointer<Integer> outputa = memOuta.read(queue);
		Pointer<Integer> outputb = memOutb.read(queue);

		// Print out the results, or check that they are correct.
		for (int i = 0; i < length; i++) {
			//System.out.println("output[" + i + "] = " + output.get(i));
			assert outputa.get(i) == length;
			assert outputb.get(i) == length + 7;
		}
	}
}
