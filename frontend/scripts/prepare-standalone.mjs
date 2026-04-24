import { cp, mkdir, readdir, rm } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';

const root = process.cwd();
const standaloneDir = path.join(root, '.next', 'standalone');
const staticSource = path.join(root, '.next', 'static');
const staticTarget = path.join(standaloneDir, '.next', 'static');
const publicSource = path.join(root, 'public');
const publicTarget = path.join(standaloneDir, 'public');

async function copyIfExists(source, target) {
  if (!existsSync(source)) return;
  await rm(target, { recursive: true, force: true });
  await mkdir(path.dirname(target), { recursive: true });
  await cp(source, target, { recursive: true });
  await removeAppleDouble(target);
}

async function removeAppleDouble(dir) {
  if (!existsSync(dir)) return;
  const entries = await readdir(dir, { withFileTypes: true });
  await Promise.all(
    entries.map(async (entry) => {
      const entryPath = path.join(dir, entry.name);
      if (entry.name.startsWith('._')) {
        await rm(entryPath, { recursive: true, force: true });
      } else if (entry.isDirectory()) {
        await removeAppleDouble(entryPath);
      }
    })
  );
}

if (!existsSync(standaloneDir)) {
  throw new Error('Missing .next/standalone. Run next build with output: "standalone" first.');
}

await copyIfExists(staticSource, staticTarget);
await copyIfExists(publicSource, publicTarget);
console.log('Prepared standalone assets.');
