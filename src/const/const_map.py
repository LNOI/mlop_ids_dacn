fmap = {'SF':0,'S0':1,'REJ':2,'RSTR':3,'RSTO':4,'SH':5 ,'S1':6 ,'S2':7,'RSTOS0':8,'S3':9 ,'OTH':10}
pmap = {'icmp':0,'tcp':1,'udp':2}

attack_type = ["dos","normal","probe","r2l","u2r"]


input = [
"duration",
"protocol_type",
"flag",
"src_bytes",
"dst_bytes",
"land",
"wrong_fragment",
"urgent",
"hot",
"num_failed_logins",
"logged_in",
"num_compromised",
"root_shell",
"su_attempted",
"num_file_creations",
"num_shells",
"num_access_files",
"is_guest_login",
"count",
"srv_count",
"serror_rate",
"rerror_rate",
"same_srv_rate",
"diff_srv_rate",
"srv_diff_host_rate",
"dst_host_count",
"dst_host_srv_count",
"dst_host_diff_srv_rate",
"dst_host_same_src_port_rate",
"dst_host_srv_diff_host_rate" ] 

# 'duration'=0
# 'protocol_type'=1
# 'flag'=0
# 'src_bytes'=181
# 'dst_bytes'=5450
# 'land'=0
# 'wrong_fragment'=0
# 'urgent'=0
# 'hot'=0
# 'num_failed_logins'=0
# 'logged_in'=1
# 'num_compromised'=0
# 'root_shell'=0
# 'su_attempted'=0
# 'num_file_creations'=0
# 'num_shells'=0
# 'num_access_files'=0
# 'is_guest_login'=0
# 'count'=8
# 'srv_count'=8
# 'serror_rate'=0.0
# 'rerror_rate'=0.0
# 'same_srv_rate'=1.0
# 'diff_srv_rate'=0.0
# 'srv_diff_host_rate'=0.0
# 'dst_host_count'=9
# 'dst_host_srv_count'=9
# 'dst_host_diff_srv_rate'=0.0
# 'dst_host_same_src_port_rate'=0.11
# 'dst_host_srv_diff_host_rate'=0.0
# 'target'=normal.
# 'Attack Type'=normal

