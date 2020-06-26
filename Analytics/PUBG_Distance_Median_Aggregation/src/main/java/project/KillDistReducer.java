package project;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class KillDistReducer extends Reducer<Text, Text, Text, Text> {
    private Map<String, Double> weap_map = new HashMap<>();

    @Override
    protected void reduce(Text key, Iterable<Text> values, Context context) {
        long counts = 0;
        double sums = 0.0;

        for (Text t : values) {
            String val = t.toString();
            //System.out.println("Key: " + key.toString() + " Val: " + val);
            sums += Double.parseDouble(val.split(":")[0]);
            counts += Long.parseLong(val.split(":")[1]);
        }
        String weapon = key.toString();
        weap_map.put(weapon, (sums / (double) counts));
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        context.write(new Text("Weapon"), new Text("Average Dist"));
        context.write(new Text(""), new Text("\n"));
        if (weap_map.size() > 0) {
            for (Map.Entry<String, Double> entry : weap_map.entrySet()) {
                context.write(new Text(entry.getKey()), new Text(String.valueOf(entry.getValue())));
            }
        }
    }
}
