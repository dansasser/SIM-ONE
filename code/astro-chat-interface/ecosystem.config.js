export default {
  apps: [
    {
      name: 'simone-chat',
      script: 'npm',
      args: 'run dev',
      cwd: '/home/user/webapp/astro-simone-chat',
      watch: false,
      env: {
        NODE_ENV: 'development',
        PORT: 3000
      },
      log_file: './logs/app.log',
      error_file: './logs/error.log',
      out_file: './logs/out.log',
      time: true,
      merge_logs: true,
      max_memory_restart: '1G'
    }
  ]
};