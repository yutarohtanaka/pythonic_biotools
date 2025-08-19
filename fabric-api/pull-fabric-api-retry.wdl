version 1.0

workflow pull_fabric_api {

    

    input {
        String run_id
        String job_id
        String FABRIC_API_LOGIN
        String FABRIC_API_PASSWORD
        String FABRIC_API_URL
        String OUTPUT_BUCKET
    }

    scatter(job_type in ["bam", "bai", "vcf", "qcmetrics"]) {
        call get_file {
            input:
                RUN_ID = run_id,
                JOB_ID = job_id,
                FILE_TYPE = job_type,
                FABRIC_API_LOGIN = FABRIC_API_LOGIN,
                FABRIC_API_PASSWORD = FABRIC_API_PASSWORD,
                FABRIC_API_URL = FABRIC_API_URL,
                OUTPUT_BUCKET = OUTPUT_BUCKET
            }
        }

}

task get_file {
    input {
        String RUN_ID
        String JOB_ID
        String FILE_TYPE
        String FABRIC_API_LOGIN
        String FABRIC_API_PASSWORD
        String FABRIC_API_URL
        String OUTPUT_BUCKET
    }

    command {
        source activate cnmf_env
        set -e

        python <<CODE

        def get_file(auth, run_id, job_id, file_type):
            """Get file by run_id, job_id and file_type"""
            # Construct request
            url = "~{FABRIC_API_URL}" + "/runs/" + run_id + "/jobs/" + job_id +"/results?key=" + file_type
            result = requests.get(url, auth=auth)
            data = result.json()
            if 'uri' in data:
                uri = result.json()['uri']
                #print(uri)
                long_file = uri.split('/')[4]
                #print (long_file)
                short_file = long_file.split('?')[0]
                print (short_file)

                response = requests.get(uri)
                #print (response)
                with open(short_file, "wb") as f:
                    f.write(response.content)
                # write the short filename to a text file so bash can pick it up
                with open("short_file.txt", "w") as f:
                    f.write(short_file)

        import os
        import requests
        from requests.auth import HTTPBasicAuth

        # Parse api credentials
        FABRIC_API_LOGIN = "~{FABRIC_API_LOGIN}"
        FABRIC_API_PASSWORD = "~{FABRIC_API_PASSWORD}"
        FABRIC_API_URL = "~{FABRIC_API_URL}"

        auth = HTTPBasicAuth(FABRIC_API_LOGIN, FABRIC_API_PASSWORD)

        run_id = "~{RUN_ID}"
        job_id = "~{JOB_ID}"
        file_type = "~{FILE_TYPE}"

        file = get_file(auth, run_id, job_id, file_type)


        CODE

        short_file=$(cat short_file.txt)
        gsutil cp "$short_file" "~{OUTPUT_BUCKET}/"
    }

    #output {
    #    File file = read_string(stdout())
    #}

    runtime {
        docker : "mparikhbroad/cnmf:latest"
        cpu : 8
        memory : "16 GB"
        disks: "local-disk 100 HDD" 
        zones : "us-central1-a"
    }

}