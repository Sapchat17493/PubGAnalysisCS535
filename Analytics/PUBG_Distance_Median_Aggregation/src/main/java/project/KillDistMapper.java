package project;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class KillDistMapper extends Mapper<LongWritable, Text, Text, Text> {
    private Map<String, List<Double>> weapon_dist;

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        super.setup(context);
        weapon_dist = new HashMap<>();
    }

    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String[] record = value.toString().split(",");
        if (record.length > 0) {
            String weapon = record[0];
            if (!weapon.equalsIgnoreCase("killed_by") && !weapon.isEmpty()) {
                if (record.length == 12) {
                    String killer_p = record[2];
                    String killer_x = "";
                    String killer_y = "";
                    if (!killer_p.isEmpty()) {
                        killer_x = record[3];
                        killer_y = record[4];
                    }
                    String victim_p = record[9];
                    String victim_x = "";
                    String victim_y = "";
                    if (!victim_p.isEmpty()) {
                        victim_x = record[10];
                        victim_y = record[11];
                    }
                    double kill_dist = 0.0;
                    try {
                        if (!killer_x.isEmpty() && !killer_y.isEmpty() && !victim_x.isEmpty() && !victim_y.isEmpty()) {
                            kill_dist = Math.sqrt(Math.pow(Double.parseDouble(victim_x) - Double.parseDouble(killer_x), 2.0) +
                                    Math.pow(Double.parseDouble(victim_y) - Double.parseDouble(killer_y), 2.0)) * 0.00001;
                        }
                    } catch (NumberFormatException nfe) {
                        nfe.printStackTrace();
                    }

                    if (kill_dist > 0.0) {
                        if (weapon_dist.containsKey(weapon)) {
                            List<Double> l = weapon_dist.get(weapon);
                            l.add(kill_dist);
                            weapon_dist.put(weapon, l);
                        } else {
                            List<Double> l = new ArrayList<>();
                            l.add(kill_dist);
                            weapon_dist.put(weapon, l);
                        }
                    }
                }
            }
        }
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        for (Map.Entry<String, List<Double>> entry : weapon_dist.entrySet()) {
            String weapon = entry.getKey();
            List<Double> list = entry.getValue();
            long count = list.size();
            double sum = sumList(list);
            //System.out.println("Writing Key: " + weapon + " Val: " + String.valueOf(sum) + ":" + String.valueOf(count));
            context.write(new Text(weapon), new Text(String.valueOf(sum) + ":" + String.valueOf(count)));
        }
    }

    private double sumList(List<Double> l) {
        double sum = 0.0;
        for (double d : l) {
            sum += d;
        }
        return sum;
    }
}
