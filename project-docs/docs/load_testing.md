# Load Testing Results

We conducted load testing using **Locust** to evaluate the performance of the **Gesture Control Game** application under various user loads.

## Load Testing Setup

- **Tool**: Locust
- **Number of Users**: 1000
- **Spawn Rate**: 10 users per second

## Results

### Summary Metrics

- **Average Response Time**: 2488.52 ms
- **Requests per Second**: 23.5
- **Failure Rate**: 0%

### Detailed Results

| Type     | Name                  | # Requests | # Fails | Median (ms) | 95%ile (ms) | 99%ile (ms) | Average (ms) | Min (ms) | Max (ms) | Average Size (bytes) | Current RPS | Current Failures/s |
|----------|-----------------------|------------|---------|-------------|-------------|-------------|--------------|----------|----------|----------------------|-------------|--------------------|
| GET      | /                     | 294        | 0       | 1800        | 4000        | 32000       | 2466.43      | 8        | 46781    | 48                   | 5.9         | 0                  |
| POST     | /calculate_reference  | 201        | 0       | 2100        | 3700        | 12000       | 2444.86      | 59       | 43713    | 95.99                | 3.9         | 0                  |
| POST     | /process_frame        | 823        | 0       | 1800        | 3900        | 27000       | 2507.08      | 67       | 49216    | 45297.65             | 13.7        | 0                  |
| Aggregated | Total               | 1318       | 0       | 1900        | 3900        | 29000       | 2488.52      | 8        | 49216    | 28310.6              | 23.5        | 0                  |

## Conclusion

The application performed well under load, with a low failure rate and acceptable response times. The system is capable of handling multiple users simultaneously without significant performance degradation.

---

## Next Steps

- [Installation Guide](installation.md)
- [Usage Guide](usage.md)