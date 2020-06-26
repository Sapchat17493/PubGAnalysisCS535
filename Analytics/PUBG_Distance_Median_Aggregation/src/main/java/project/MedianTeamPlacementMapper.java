package project;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

public class MedianTeamPlacementMapper extends Mapper<LongWritable, Text, Text, DoubleWritable> {

    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String[] record = value.toString().split(",");
        if (record.length > 0) {
            if (!record[0].equalsIgnoreCase("date")) {
                if (record.length == 15) {
                    context.write(new Text(record[14]), new DoubleWritable(Double.parseDouble(record[12])));
                }
            }
        }
    }

}
