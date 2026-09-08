/** Restore one explicitly supplied existing Android task without a new Intent.
 * Runs as an adb shell app_process, does not alter process priority or policy. */
public final class E1BTaskFront {
    public static void main(String[] args) throws Exception {
        if (args.length != 1) throw new IllegalArgumentException("expected positive task id");
        int task = Integer.parseInt(args[0]);
        if (task <= 0) throw new IllegalArgumentException("invalid task id");
        Class<?> manager = Class.forName("android.app.ActivityTaskManager");
        Object service = manager.getMethod("getService").invoke(null);
        Class<?> api = Class.forName("android.app.IActivityTaskManager");
        java.lang.reflect.Method method = api.getMethod("startActivityFromRecents", int.class, Class.forName("android.os.Bundle"));
        long begin = System.nanoTime();
        Object result = method.invoke(service, task, null);
        long end = System.nanoTime();
        System.out.println("TASK="+task+" START_US="+(begin/1000)+" END_US="+(end/1000)+" RESULT="+result);
    }
}
