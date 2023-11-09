# Intrusion Detection System

Project comprised of ARCADE and MPI code.

# Features

| Name                         | Description                                                                                       |
|------------------------------|---------------------------------------------------------------------------------------------------|
| **UDP Features**             |                                                                                                   |
| `n_packets`                  | Number of packets in the flow.                                                                    |
| `duration`                   | Time duration of the flow.                                                                        |
| `max_ttl`                    | Maximum time-to-live (TTL) value from the packets in the flow.                                    |
| `min_ttl`                    | Minimum TTL value from the packets in the flow.                                                   |
| `total_bytes`                | Total bytes of the packets in the flow, including IP headers, transport headers, and payload data. |
| `n_empty_trans_header`       | Count of packets with empty transport header.                                                  |
| `n_empty_trans_header_per`   | Percentage of packets with empty transport header.                                             |
| `n_empty_trans_payload`      | Count of packets with empty transport payload.                                                |
| `n_empty_trans_payload_per`  | Pecentage of packets with empty transport payload.                                            |
| `bytes_sec`                  | Average number of total bytes per second.                                                                                                                                    |
| `packets_sec`                | Average number of packets per second.                                                                                                                                        |
| `idle_time`                  | Total idle time during the flow duration, calculated as time when inter-arrival time of packets is more than 2 seconds.                                                      |
| `idle_per`                   | Percentage of idle time in the flow duration.                                                                                                                                |
| `idle_bef_tw`                | Idle time before the time window, if any.                                                                                                                                    |
| `active_time`                | Active time during the flow duration, calculated as duration minus idle time.                                                                                                |
| `active_per`                 | Percentage of active time in the flow duration.                                                                                                                              |
| `bytes_active_sec`           | Average number of bytes per second during active time.                                                                                                                       |
| `packets_active_sec`         | Average number of packets per second during active time.                                                                                                                     |
| `iat`                        | Statistical feature object of inter-arrival time of packets, calculated as the time difference between consecutive packets.                                                  |
| `ip_header`                  | Statistical feature object of IP header sizes, including metrics like min, max, mean, standard deviation, variance, skewness, kurtosis, percentile 25, median, percentile 75. |
| `ip_payload`                 | Statistical feature object of IP payload sizes (transport header and payload), , similar to `ip_header`.                                                                     |
| `trans_header`               | Statistical feature object of transport header sizes, similar to `ip_header`.                                                                                                |
| `trans_payload`              | Statistical feature object of payload siz, similar to `ip_header`. es, similar to `ip_header`.                                                                               |
| `pkt_time`                   | Average flight time of one packet computed as duration divided by number of packets.                                                                                         |
| `pkt_size`                   | Average size of one packet computed as total bytes divided by number of packets.                                                                                             |
| `n_subflows`                 | Number of active intervals in the flow (i.e. not interrupted by idle time).                                                                                                  |
| `subflow_bytes`              | Statistical feature object of subflow bytes computed as the bytes of packets active interval in the flow.                                                                    |
| `subflow_pkts`               | Statistical feature object of number of subflow packets computed as the number of packets in each active interval in the flow.                                               |
| `subflow_time`               | Statistical feature object of the duration of the subflow computed as the number of packets in each active interval in the flow.                                             |
| **TCP Features**             |                                                                                                                                                                              |
| `service`                    | The service associated with the source port.                                                                                                                                 |
| `out_of_order`               | Count of out-of-order TCP packets, calculated by comparing the sequence numbers of consecutive packets.                                                                      |
| `urgent_bytes`               | The total bytes of urgent data sent computed by summing the urgent pointer offset value in URG packets.                                                                      |
| `urgent_bytes_per`           | Percentage of total bytes that contain urgent bytes.                                                                                                                         |
| `max_req_segment_size`       | Maximum requested segment size in the flow.                                                                                                                                  |
| `tcp_window_scale`           | TCP window scale size, optionally set in the TCP header.                                                                                                                     |
| `n_tcp_zero_window`          | Count of times a zero receive window was advertised computed as TCP window scale set to zero.                                                                                |
| `n_tcp_zero_window_per`      | Percentage of zero receive window packets advertised.                                                                                                                        |
| `n_repeated_seq_numbers`     | Count of packets with duplicate sequence numbers.                                                                                                                            |
| `n_repeated_seq_numbers_per` | Percentage of packets with duplicate sequence numbers.                                                                                                                       |
| `n_repeated_ack_numbers`     | Count of packets with duplicate acknowledgement numbers.                                                                                                                     |
| `n_repeated_ack_numbers_per` | Percentage of packets with duplicate acknowledgement numbers.                                                                                                                |
| `pure_acks_sent`             | Count of ACK packets without transport payload (just the TCP header) and without SYN/FIN/RST flags set.                                                                      |
| `pure_acks_sent_per`         | Percentage of packets that are pure_ack packets.                                                                                                                             |
| `n_fin`                      | Count of packets with the FIN flag set.                                                                                                                                      |
| `n_syn`                      | Count of packets with the SYN flag set.                                                                                                                                      |
| `n_rst`                      | Count of packets with the RST flag set.                                                                                                                                      |
| `n_psh`                      | Count of packets with the PSH flag set.                                                                                                                                      |
| `n_ack`                      | Count of packets with the ACK flag set.                                                                                                                                      |
| `n_urg`                      | Count of packets with the URG flag set.                                                                                                                                      |
| `n_ece`                      | Count of packets with the ECE flag set.                                                                                                                                      |
| `n_cwr`                      | Count of packets with the CWR flag set.                                                                                                                                      |
| `n_ns`                       | Count of packets with the NS flag set.                                                                                                                                       |
| `n_sack`                     | Count of packets with the SACK option set.                                                                                                                                   |
| `tcp_window_size`            | Statistical feature object of the TCP window size in the packet header field.                                                                                                |
| **Biflow Features**          |                                                                                                                                                                              |
| `down_up_ratio`              | Download and upload ratio computed as number of packets in the backward direction divided by number of packets in the forward direction.                                     |
| `rtt`                        | Initial round trip time determined by the difference in the timestamps of the SYN packets in the TCP Three Way Handshake.                                                    |
| `mean_rtt`                   | Average initial round trip time of the forward and backward flows determined by the difference in the timestamps of the SYN packets in the TCP Three Way Handshake.          |


| **Current State**         | **Triggering Event** | **Next State**                |
|---------------------------|----------------------|-------------------------------|
| IDLE                      | PERIODIC_SCAN        | PERFORMING_SCAN               |
| IDLE                      | EXT_SWITCH_EVENT     | SWITCHING_CHANNEL             |
| IDLE                      | EXT_DATA_REQUEST     | REPORTING_DATA_CH             |
| PERFORMING_SCAN           | NO_JAM_DETECTED      | IDLE                          |
| PERFORMING_SCAN           | JAM_DETECTED         | SENDING_JAM_ALERT             |
| PERFORMING_SCAN           | EXT_DATA_REQUEST     | REPORTING_DATA_CH             |
| PERFORMING_SCAN           | EXT_SWITCH_EVENT     | SWITCHING_CHANNEL             |
| PERFORMING_SCAN           | UPDATED_SCAN         | REPORTING_DATA_CH             |
| SENDING_JAM_ALERT         | JAM_ALERT_SENT       | IDLE                          |
| SENDING_JAM_ALERT         | EXT_DATA_REQUEST     | REPORTING_DATA_CH             |
| SENDING_JAM_ALERT         | EXT_SWITCH_EVENT     | SWITCHING_CHANNEL             |
| REPORTING_DATA_CH         | DATA_REPORT_SENT     | IDLE                          |
| REPORTING_DATA_CH         | EXT_SWITCH_EVENT     | SWITCHING_CHANNEL             |
| REPORTING_DATA_CH         | EXPIRED_SCAN         | PERFORMING_SCAN               |
| SWITCHING_CHANNEL         | SWITCHED             | RESETTING                     |
| SWITCHING_CHANNEL         | SWITCH_FAILED        | RECOVERING_SWITCH_ERROR       |
| RECOVERING_SWITCH_ERROR   | PERIODIC_SWITCH      | SWITCHING_CHANNEL             |
| RECOVERING_SWITCH_ERROR   | EXT_SWITCH_EVENT     | SWITCHING_CHANNEL             |
| RESETTING                 | RESET                | IDLE                          |