package project;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class MedianTeamSizeReducer extends Reducer<Text, DoubleWritable, Text, Text> {

    @Override
    protected void reduce(Text key, Iterable<DoubleWritable> values, Context context) throws IOException, InterruptedException {
        List<Double> vals = new ArrayList<>();
        for (DoubleWritable v : values) {
            vals.add(v.get());
        }

        Collections.sort(vals);
        String out = key.toString() + "," + String.valueOf(vals.get((int) (vals.size() / 2)));


        //context.write(key, new Text(String.valueOf(vals.get((int) (vals.size() / 2)))));
        context.write(new Text(out), new Text(""));
    }
}
